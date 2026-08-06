// Centralised stb_image / stb_image_write implementations.
//
// Both headers are included with their *_IMPLEMENTATION defines exactly once
// in the whole project.  Every target that needs stb (trellis2, rmbg via
// trellis2, examples) links the stb_impl object, avoiding multiple-definition
// errors that arise when several translation units each compile the
// implementations.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
