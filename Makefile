# the compiler: gcc for C program, define as g++ for C++ 
CC = clang++

# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings
#  -v     - verbose readout, disable if not actively debugging package bullcrap
export CPATH=/usr/include/hdf5/serial/
CFLAGS  = -g -Wall -fsanitize=address -fsanitize=leak -lGL 
H5FLAGS = -I/home/morgan/miniconda3/include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/morgan/miniconda3/include -fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/morgan/miniconda3/include -fdebug-prefix-map=/tmp/build/80754af9/hdf5_1593120246797/work=/usr/local/src/conda/hdf5-1.10.6 -fdebug-prefix-map=/home/morgan/miniconda3=/usr/local/src/conda-prefix -L/home/morgan/miniconda3/lib /home/morgan/miniconda3/lib/libhdf5_hl_cpp.a /home/morgan/miniconda3/lib/libhdf5_cpp.a /home/morgan/miniconda3/lib/libhdf5_hl.a /home/morgan/miniconda3/lib/libhdf5.a -L/home/morgan/miniconda3/lib -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,-rpath,/home/morgan/miniconda3/lib -Wl,-rpath-link,/home/morgan/miniconda3/lib -L/home/morgan/miniconda3/lib -lrt -lpthread -lz -ldl -lm -Wl,-rpath -Wl,/home/morgan/miniconda3/lib


# The build target 
TARGET = main
#TARGET = 

all: $(TARGET) run

$(TARGET): $(TARGET).cpp
	$(CC) $(CFLAGS) $(H5FLAGS) -o $(TARGET) $(TARGET).cpp
#imgui/*.cpp put back to line above if gonna use

run:
	./$(TARGET)

clean:
	$(RM) $(TARGET)

rebuild:
	$(RM) $(TARGET)
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cpp
	./$(TARGET)

