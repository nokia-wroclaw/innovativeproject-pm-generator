<!--
  Tooltip — thin wrapper around Reka UI's TooltipRoot.

  Usage:
    <Tooltip content="Run ID: abc-123">
      <button>...</button>
    </Tooltip>
-->
<template>
  <TooltipProvider :delay-duration="200">
    <TooltipRoot>
      <TooltipTrigger as-child>
        <slot />
      </TooltipTrigger>
      <TooltipPortal>
        <TooltipContent
          :side="side"
          :side-offset="6"
          class="z-[1100] max-w-xs rounded-md bg-slate-900 px-2.5 py-1.5 text-xs font-medium leading-snug text-white shadow-md
                 whitespace-normal data-[state=delayed-open]:animate-in data-[state=delayed-open]:fade-in-0
                 data-[state=closed]:animate-out data-[state=closed]:fade-out-0"
        >
          {{ content }}
        </TooltipContent>
      </TooltipPortal>
    </TooltipRoot>
  </TooltipProvider>
</template>

<script setup>
import {
  TooltipProvider,
  TooltipRoot,
  TooltipTrigger,
  TooltipPortal,
  TooltipContent,
} from 'reka-ui';

defineProps({
  content: { type: String, required: true },
  /** @type {'top' | 'right' | 'bottom' | 'left'} */
  side: { type: String, default: 'top' },
});
</script>
