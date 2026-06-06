<template>
  <span class="group/hint relative inline-flex">
    <button
      type="button"
      tabindex="-1"
      class="inline-flex shrink-0 text-fg-subtle transition-colors hover:text-fg"
      :aria-label="text"
      @mousedown.prevent
    >
      <CircleHelp :size="14" />
    </button>
    <span
      role="tooltip"
      class="pointer-events-none absolute left-[calc(100%+6px)] top-1/2 z-[1100] hidden w-max max-w-xs -translate-y-1/2 rounded-md bg-slate-900 px-2.5 py-1.5 text-xs font-medium leading-snug text-white shadow-md group-hover/hint:block"
    >
      {{ text }}
    </span>
  </span>
</template>

<script setup>
import { CircleHelp } from 'lucide-vue-next';

defineProps({
  text: { type: String, required: true },
});
</script>
