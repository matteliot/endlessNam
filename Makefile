# endlessName: NAM amp/cab rigs for the Polyend Endless.
#
#   make                         build the default rig -> build/<rig>/<rig>.endl
#   make RIG=rigs/other.json     build another rig
#   make deploy                  build + copy the .endl to the mounted pedal
#   make host                    native test renderer for the current rig
#   make render IN=di.wav        render a WAV through the rig on this computer
#   make test                    check the engine against NeuralAmpModelerCore
#   make clean

RIG ?= rigs/soldano_rip.json
RIG_NAME := $(basename $(notdir $(RIG)))
ENDLESS_VOLUME ?= /Volumes/ENDLESS

SDK := third_party/FxPatchSDK
NAMCORE := third_party/NeuralAmpModelerCore
PYTHON ?= python3

# Find the Arm GNU toolchain: PATH first, then the macOS installer location,
# then an unpacked copy in ~/opt/arm-gnu-toolchain-*.
ifeq ($(origin TOOLCHAIN), undefined)
  ifneq ($(shell command -v arm-none-eabi-gcc 2>/dev/null),)
    TOOLCHAIN := arm-none-eabi-
  else
    TOOLCHAIN_DIR := $(lastword $(sort $(wildcard /Applications/ArmGNUToolchain/*/arm-none-eabi/bin/) \
                                       $(wildcard $(HOME)/opt/arm-gnu-toolchain-*/bin/)))
    TOOLCHAIN := $(TOOLCHAIN_DIR)arm-none-eabi-
  endif
endif

BUILD := build/$(RIG_NAME)
GEN := $(BUILD)/generated
RIG_HEADER := $(GEN)/rig_data.h
ENDL := $(BUILD)/$(RIG_NAME).endl
ELF := $(BUILD)/$(RIG_NAME).elf

# ---------------------------------------------------------------- pedal build
# Flags follow the SDK Makefile (third_party/FxPatchSDK/Makefile).
CC := $(TOOLCHAIN)gcc
CXX := $(TOOLCHAIN)g++
OBJCOPY := $(TOOLCHAIN)objcopy
SIZE := $(TOOLCHAIN)size

COMMON_FLAGS := -Wall -Wextra -Werror -fno-builtin -fno-common -ffunction-sections -fdata-sections \
                -fsingle-precision-constant -g -mfloat-abi=hard -mthumb \
                -fstack-usage -specs=nano.specs -Wdouble-promotion -mcpu=cortex-m7 -mfpu=fpv5-sp-d16 \
                -mfp16-format=ieee -O3 -DNDEBUG
CFLAGS := $(COMMON_FLAGS) -std=c11
CXXFLAGS := $(COMMON_FLAGS) -std=c++20 -includecstdint -felide-constructors -Wno-psabi -fno-exceptions -fno-rtti \
            -Isrc -I$(SDK)/source -I$(GEN)
LDFLAGS := -nostartfiles --specs=nosys.specs -Wl,-gc-sections -T $(SDK)/internal/patch_imx.ld \
           -Wl,--defsym=PATCH_LOAD_ADDR=0x80000000 -Wl,--defsym=end=__patch_bss_end -Wl,-Map=$(BUILD)/$(RIG_NAME).map
LIBS := -Wl,--start-group -lstdc++ -lc -lm -lgcc -Wl,--end-group

OBJ := $(BUILD)/obj/sdk/patch_main.o $(BUILD)/obj/sdk/PatchCppWrapper.o $(BUILD)/obj/PatchImpl.o

all: $(ENDL)

$(RIG_HEADER): $(RIG) tools/build_rig.py
	@mkdir -p $(GEN)
	$(PYTHON) tools/build_rig.py $(RIG) -o $@

$(BUILD)/obj/sdk/%.o: $(SDK)/internal/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/obj/sdk/%.o: $(SDK)/internal/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/PatchImpl.o: src/PatchImpl.cpp src/rig.h src/nam_a2.h src/fir.h $(RIG_HEADER)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ELF): $(OBJ) $(SDK)/internal/patch_imx.ld
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS) $(LIBS)
	@$(SIZE) $@

$(ENDL): $(ELF)
	$(OBJCOPY) -O binary $< $@
	@echo "built $@ ($$(wc -c < $@ | tr -d ' ') bytes)"

deploy: $(ENDL)
	@test -d "$(ENDLESS_VOLUME)" || { echo "Endless not mounted at $(ENDLESS_VOLUME) (set ENDLESS_VOLUME=...)"; exit 1; }
	cp $(ENDL) "$(ENDLESS_VOLUME)/"
	@echo "copied $(notdir $(ENDL)) to $(ENDLESS_VOLUME)"

# ---------------------------------------------------------------- host tools
HOSTCXX ?= clang++
HOST_RENDER := $(BUILD)/host/render

host: $(HOST_RENDER)

$(HOST_RENDER): host/render.cpp src/rig.h src/nam_a2.h src/fir.h $(RIG_HEADER)
	@mkdir -p $(dir $@)
	$(HOSTCXX) -std=c++17 -O2 -Wall -Wextra -Isrc -I$(GEN) $< -o $@

IN ?= host/testdata/test_in.wav
OUT ?= $(BUILD)/render_out.wav
render: $(HOST_RENDER)
	$(HOST_RENDER) $(IN) $(OUT) $(ARGS)
	@echo "wrote $(OUT)"

# Reference renderer built from NeuralAmpModelerCore, for `make test`.
NAM_REF := build/nam-ref/render
$(NAM_REF):
	@mkdir -p $(dir $@)
	$(HOSTCXX) -std=c++2a -O2 -DNAM_ENABLE_A2_FAST -DJSON_HAS_FILESYSTEM=0 -DJSON_HAS_EXPERIMENTAL_FILESYSTEM=0 \
	  -I$(NAMCORE) -I$(NAMCORE)/Dependencies/AudioDSPTools/dsp -I$(NAMCORE)/Dependencies/eigen \
	  -I$(NAMCORE)/Dependencies/nlohmann $(NAMCORE)/NAM/*.cpp $(NAMCORE)/NAM/wavenet/*.cpp \
	  $(NAMCORE)/Dependencies/AudioDSPTools/dsp/wav.cpp $(NAMCORE)/tools/render.cpp -o $@

test: $(NAM_REF)
	host/run_tests.sh $(NAM_REF)

clean:
	rm -rf build

.PHONY: all deploy host render test clean
