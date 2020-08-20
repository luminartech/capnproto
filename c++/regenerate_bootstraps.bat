@echo off

set FOUND=0
set BINPATH=%CD%\build\src\capnp
IF EXIST %BINPATH% (
    IF EXIST %BINPATH%"\Release\capnp.exe" (
        set BINPATH=%BINPATH%\Release
        set FOUND=1
    )

    IF EXIST %BINPATH%"\Debug\capnp.exe" (
        set BINPATH=%BINPATH%\Debug
        set FOUND=1
    )
)

IF %FOUND%==0 (
    echo "capnp.exe is not found"
    exit /B
)

set ORIG_PATH=%PATH%
set PATH=%BINPATH%;%CD%;%PATH%

%BINPATH%\capnp compile -Isrc --no-standard-import --src-prefix=src -oc++:src ^
    src/capnp/c++.capnp src/capnp/schema.capnp ^
    src/capnp/compiler/lexer.capnp src/capnp/compiler/grammar.capnp ^
    src/capnp/rpc.capnp src/capnp/rpc-twoparty.capnp src/capnp/persistent.capnp ^
    src/capnp/compat/json.capnp

set PATH=%ORIG_PATH%