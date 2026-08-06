// Encode an image into the TRELLIS.2 DINOv3 conditioning tensor (.dinodata),
// replacing the external dump_dinodata.py:
//
//   image (PNG/JPG/WebP; solid black/white backgrounds are removed automatically)
//     -> background cleanup -> alpha bbox crop, premultiply, LANCZOS 512
//     -> DINOv3 ViT-L/16 -> [1, 1029, 1024] cond -> .dinodata
//
// usage: dino_encode <dino.gguf> <image> [out.dinodata] [--size N] [--pre out.png]
//
#include "trellis2.h"

// stb_image and stb_image_write implementations are provided by libtrellis2
// (mesh_export.cpp / trellis2_capi.cpp). We only need the declarations here.
#include "stb_image.h"
#include "stb_image_write.h"

#ifdef TRELLIS2_HAVE_WEBP
#include <webp/decode.h>
#endif

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <dino.gguf> <image> [out.dinodata] [--size N] [--pre out.png]\n",
            argv[0]);
        return 2;
    }
    const std::string gguf_path = argv[1];
    const std::string img_path  = argv[2];
    std::string out_path = "cond.dinodata";
    std::string pre_path;
    int size = 512;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--pre") == 0 && i + 1 < argc) {
            pre_path = argv[++i];
        } else {
            out_path = argv[i];
        }
    }

    // Read the image file into memory so we can handle WebP as well as
    // stb_image formats (JPEG/PNG/TGA/BMP/GIF/HDR/PIC/PNM).
    std::ifstream ifs(img_path, std::ios::binary);
    if (!ifs) {
        std::fprintf(stderr, "cannot open %s\n", img_path.c_str());
        return 1;
    }
    ifs.seekg(0, std::ios::end);
    auto fsize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (fsize <= 0 || fsize > 256 * 1024 * 1024) {
        std::fprintf(stderr, "file %s is empty or too large\n", img_path.c_str());
        return 1;
    }
    std::vector<uint8_t> file_buf((size_t) fsize);
    ifs.read(reinterpret_cast<char *>(file_buf.data()), fsize);
    ifs.close();

    int w = 0, h = 0, comp = 0;
    unsigned char * pixels = nullptr;
    bool from_webp = false;

#ifdef TRELLIS2_HAVE_WEBP
    // Detect WebP "RIFF....WEBP" signature.
    if (file_buf.size() >= 12
        && file_buf[0] == 'R' && file_buf[1] == 'I'
        && file_buf[2] == 'F' && file_buf[3] == 'F'
        && file_buf[8] == 'W' && file_buf[9] == 'E'
        && file_buf[10] == 'B' && file_buf[11] == 'P') {
        from_webp = true;
        int wp = 0, hp = 0;
        if (!WebPGetInfo(file_buf.data(), file_buf.size(), &wp, &hp)) {
            std::fprintf(stderr, "WebP probe failed for %s\n", img_path.c_str());
            return 1;
        }
        w = wp; h = hp;
        pixels = WebPDecodeRGBA(file_buf.data(), file_buf.size(), &wp, &hp);
        if (!pixels) {
            std::fprintf(stderr, "WebP decode failed for %s\n", img_path.c_str());
            return 1;
        }
        comp = 4;
    } else
#endif
    {
        pixels = stbi_load_from_memory(file_buf.data(), (int) file_buf.size(),
                                        &w, &h, &comp, 4);
        if (!pixels) {
            std::fprintf(stderr, "failed to decode image %s: %s\n",
                         img_path.c_str(), stbi_failure_reason());
            return 1;
        }
    }

    const int removed = trellis2_remove_solid_background_rgba(
        pixels, w, h, TRELLIS2_BACKGROUND_AUTO);
    std::printf("image  : %s  %dx%d (%d channels%s, background pixels changed: %d)\n",
                img_path.c_str(), w, h, comp,
                from_webp ? ", webp" : "", removed);

    std::string err;
    std::vector<uint8_t> rgb;
    if (!trellis2_preprocess_rgba(pixels, w, h, size, rgb, &err)) {
        std::fprintf(stderr, "preprocess failed: %s\n", err.c_str());
#ifdef TRELLIS2_HAVE_WEBP
        if (from_webp) WebPFree(pixels); else
#endif
        stbi_image_free(pixels);
        return 1;
    }
#ifdef TRELLIS2_HAVE_WEBP
    if (from_webp) WebPFree(pixels); else
#endif
    stbi_image_free(pixels);

    if (!pre_path.empty()) {
        stbi_write_png(pre_path.c_str(), size, size, 3, rgb.data(), size * 3);
        std::printf("wrote  : %s (preprocessed %dx%d RGB)\n", pre_path.c_str(), size, size);
    }

    trellis2_dino_model * model = trellis2_dino_load(gguf_path, true, &err);
    if (!model) {
        std::fprintf(stderr, "model load failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("model  : %s (backend %s)\n", gguf_path.c_str(),
                trellis2_dino_backend_name(model));

    trellis2_dino_cond cond;
    if (!trellis2_dino_encode_rgb(model, rgb.data(), size, cond, &err)) {
        std::fprintf(stderr, "encode failed: %s\n", err.c_str());
        trellis2_dino_free(model);
        return 1;
    }
    trellis2_dino_free(model);

    const trellis2_dino_fingerprint fp = trellis2_dino_fingerprints(cond);
    std::printf("cond   : [1, %lld, %lld]  min=%.4f max=%.4f mean=%.6f l2=%.4f\n",
                (long long) cond.tokens(), (long long) cond.channels(),
                fp.vmin, fp.vmax, fp.mean, fp.l2);

    if (!trellis2_save_dinodata(out_path, cond, &err)) {
        std::fprintf(stderr, "save failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("wrote  : %s (%zu floats)\n", out_path.c_str(), cond.count());
    return 0;
}
