set INCLUDE_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\include"
set LIB_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\lib\x64"

cl /I %INCLUDE_PATH% %1 /link /LIBPATH:%LIB_PATH% cudart.lib cuda.lib


