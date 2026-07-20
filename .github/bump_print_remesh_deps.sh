#!/usr/bin/env bash
# Update the coupled version + checksum pins in fetch_print_remesh_deps.sh.
set -euo pipefail

FILE=${1:-scripts/fetch_print_remesh_deps.sh}
current_cgal=$(grep -m1 '^CGAL_VERSION=' "$FILE" | cut -d= -f2)
current_boost=$(grep -m1 '^BOOST_VERSION=' "$FILE" | cut -d= -f2)
if [[ -z "$current_cgal" || -z "$current_boost" ]]; then
	echo "Could not read current dependency versions from $FILE" >&2
	exit 1
fi

read -r new_cgal cgal_sha < <(
	curl -fsSL -H 'Accept: application/vnd.github+json' \
		https://api.github.com/repos/CGAL/cgal/releases/latest |
		python3 -c '
import json, sys
release = json.load(sys.stdin)
version = release["tag_name"].removeprefix("v")
name = f"CGAL-{version}-library.zip"
asset = next(a for a in release["assets"] if a["name"] == name)
digest = asset["digest"]
if not digest.startswith("sha256:"):
    raise SystemExit(f"missing SHA-256 digest for {name}")
print(version, digest.removeprefix("sha256:"))
'
)

new_boost=$(
	curl -fsSL https://archives.boost.io/release/ |
		python3 -c '
import re, sys
versions = re.findall(r"href=\"([0-9]+\.[0-9]+\.[0-9]+)/\"", sys.stdin.read())
if not versions:
    raise SystemExit("no stable Boost releases found")
print(max(versions, key=lambda value: tuple(map(int, value.split(".")))))
'
)
if [[ ! "$new_cgal" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ || ! "$new_boost" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
	echo "Invalid release versions: CGAL=$new_cgal Boost=$new_boost" >&2
	exit 1
fi

boost_archive="boost_${new_boost//./_}.tar.gz"
boost_sha=$(
	curl -fsSL "https://archives.boost.io/release/$new_boost/source/$boost_archive.json" |
		python3 -c 'import json, sys; print(json.load(sys.stdin)["sha256"])'
)
if [[ ! "$cgal_sha" =~ ^[0-9a-f]{64}$ || ! "$boost_sha" =~ ^[0-9a-f]{64}$ ]]; then
	echo "Invalid SHA-256 digests for CGAL or Boost" >&2
	exit 1
fi

sed -i \
	-e "s/^CGAL_VERSION=.*/CGAL_VERSION=$new_cgal/" \
	-e "s/^CGAL_SHA256=.*/CGAL_SHA256=$cgal_sha/" \
	-e "s/^BOOST_VERSION=.*/BOOST_VERSION=$new_boost/" \
	-e "s/^BOOST_SHA256=.*/BOOST_SHA256=$boost_sha/" \
	"$FILE"

{
	echo "CGAL $current_cgal -> $new_cgal: https://github.com/CGAL/cgal/releases/tag/v$new_cgal"
	echo
	echo "Boost $current_boost -> $new_boost: https://www.boost.org/releases/$new_boost/"
} > print_remesh_deps_message.txt
printf '%s\n' "CGAL-$new_cgal/Boost-$new_boost" > print_remesh_deps_version.txt
