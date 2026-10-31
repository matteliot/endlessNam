/* Patch image layout for the Endless. Same sections and symbols as the SDK's
 * third_party/FxPatchSDK/internal/patch_imx.ld; the header itself is defined
 * in Rust (src/abi.rs) instead of patch_main.c. */

_PATCH_BASE = DEFINED(PATCH_LOAD_ADDR) ? PATCH_LOAD_ADDR : 0x80000000;

MEMORY
{
  RAM (rxw) : ORIGIN = _PATCH_BASE, LENGTH = 512K
}

ENTRY(patch_header)

SECTIONS
{
  .patch_header ORIGIN(RAM) : ALIGN(4)
  {
    KEEP(*(.patch_header))
  } > RAM

  .patch_image : ALIGN(4)
  {
    __patch_start = .;
    *(.text .text.*)
    *(.rodata .rodata.*)
    . = ALIGN(4);
    *(.data .data.*)
    . = ALIGN(4);
    __patch_end = .;
  } > RAM

  .patch_bss (NOLOAD) : ALIGN(4)
  {
    __patch_bss_start = .;
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(4);
    __patch_bss_end = .;
  } > RAM

  /DISCARD/ :
  {
    *(.comment*)
    *(.note*)
    /* Unwind tables: unused with panic = "abort". */
    *(.ARM.exidx .ARM.exidx.*)
    *(.ARM.extab .ARM.extab.*)
  }

  /* Header fields (absolute values, read via the symbol's address). */
  patch_image_size = __patch_end - __patch_start;
  patch_bss_size = __patch_bss_end - __patch_bss_start;
}
