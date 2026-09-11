<script setup lang="ts">
/** Wordmark inline pelo mesmo motivo do BrandMark: herda currentColor. Largura
 *  derivada do viewBox (494/84) para nao deformar em tamanho nenhum. */
import { computed } from "vue";

const props = withDefaults(defineProps<{ height?: number }>(), { height: 15 });
const width = computed(() => Math.round((props.height * 494) / 84));
</script>

<template>
  <svg
    viewBox="0 0 494 84"
    :width="width"
    :height="height"
    fill="none"
    aria-hidden="true"
    focusable="false"
    class="shrink-0"
  >
    <g stroke="currentColor" stroke-width="9" stroke-linecap="round" stroke-linejoin="round">
      <path d="M6 78 47.57 6 89.14 78M20 54H75" />
      <path transform="translate(113)" d="M49 6H6V78H49M6 42H40" />
      <path transform="translate(188)" d="M4 6H62M33 6V78" />
      <path transform="translate(270)" d="M6 6V78M61 6V78M6 42H61" />
      <path transform="translate(357)" d="M49 6H6V78H49M6 42H40" />
      <path transform="translate(432)" d="M6 78V6H40L54 14V34L40 42H6M30 42 50.78 78" />
    </g>
  </svg>
</template>
