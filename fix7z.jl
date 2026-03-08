# Workaround for 7zip v25 incompatibility with Julia 1.11-rc3 on Fedora 43.
# The bundled 7z (symlinked to /usr/bin/7z) fails to decompress .tar.gz
# artifacts when piped via Julia's process system.
# This patches Pkg.PlatformEngines.exe7z to use a gzip-based wrapper.
import Pkg.PlatformEngines
PlatformEngines.exe7z() = `$(homedir())/.local/bin/7z-julia-compat`
