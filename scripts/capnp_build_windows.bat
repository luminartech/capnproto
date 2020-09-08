@echo off

set buildSystemGenerator=Visual Studio 16 2019
set buildType=RELEASE
set buildOptions=
set buildArch=x64
set buildOS=Windows10
set toolchain="%~dp0/../../cmake/toolchains/x86_64/windows/msvc/toolchain_x86_64_windows10_msvc.cmake"
set installFolder="%~dp0/../../install/x86_64/windows/msvc"
set cmakePrefixPath="%~dp0/../../install/x86_64/windows/msvc"

set buildSolution=true
set buildTests=true

:getopts
    if /I "%1" == "-h" goto usage
    if /I "%1" == "-s" set buildSystemGenerator=%2 & shift
    if /I "%1" == "-t" set buildType=%2 & shift
    if /I "%1" == "-o" set buildOptions="%buildOptions% %2" & shift
    if /I "%1" == "-a" set buildArch=%2 & shift
    if /I "%1" == "-l" set buildOS=%2 & shift
    if /I "%1" == "-p" set toolchain=%2 & shift
    if /I "%1" == "-c" set buildSolution=false
    shift
if not "%1" == "" goto getopts

rmdir /s /q build

mkdir build

set lumpdkBuildArch=%buildArch%
if ["%buildArch%"] == ["x64"] set lumpdkBuildArch=x86_64

cmake -Bbuild -H. ^
 -A"%buildArch%" ^
 -G"%buildSystemGenerator%" ^
 -DCMAKE_BUILD_TYPE=%buildType% ^
 -DBUILD_ARCH=%lumpdkBuildArch% ^
 -DBUILD_OS=%buildOS% ^
 -DBUILD_TESTING=%buildTests% ^
 %buildOptions% ^
 -DCMAKE_TOOLCHAIN_FILE=%toolchain% ^
 -DCMAKE_PREFIX_PATH:PATH=%cmakePrefixPath% ^
 -DCMAKE_INSTALL_PREFIX:PATH=%cmakePrefixPath% ^
 -DBUILD_SHARED_LIBS=ON || goto fail

cmake --build ./build --config %buildType% --target install || goto fail

:succeed
popd
echo Build successful!
exit /b 0

:usage
echo Usage:
echo build_lumpdk_windows.bat [OPTIONS...]
echo    -h  Show this help menu
echo    -s  Set build system generator [default: "Visual Studio 16 2019"]
echo    -t  Set build type [default: RELEASE, options: DEBUG^|RELEASE]
echo    -o  Add cmake options [default:none, example: -DCOVERAGE=ON]
echo    -d  Dirty build (i.e., do not delete build space)
echo    -a  Set build architecture (i.e., generator platform) [default: x64, options: Win32^|x64^|ARM^|ARM64 for Visual Studio 2017 and 2019]
echo    -a  Set operating system [default: Windows10]
echo    -p  Set toolchain file [default: cmake/toolchains/x86_64/windows/msvc/toolchain_x86_64_windows10_msvc.cmake]
exit /b 0

:fail
popd
echo BUILD FAILED
exit /b 1