<!--
  Sheet — slide-in side panel.

  Thin wrapper around Reka UI's DialogRoot. The `side="right"` variant slides
  in from the right edge (used for TaskDetailsSheet); the contract mandates
  focus-trap, ESC-to-close, and a backdrop with click-to-dismiss — all of
  which DialogRoot provides for free.

  Usage:
    <Sheet v-model:open="open" side="right">
      <template #title>Task details</template>
      <template #description>generate_synthetic_pm · transform_table</template>
      ...content...
    </Sheet>
-->
<template>
  <DialogRoot :open="open" @update:open="$emit('update:open', $event)">
    <DialogPortal>
      <DialogOverlay
        class="fixed inset-0 z-40 bg-slate-900/30 backdrop-blur-[2px]
               data-[state=open]:animate-in data-[state=open]:fade-in-0
               data-[state=closed]:animate-out data-[state=closed]:fade-out-0"
      />
      <DialogContent
        :class="cn(
          'fixed z-50 flex flex-col bg-surface shadow-xl',
          'focus:outline-none',
          sideClasses[side],
        )"
      >
        <div class="flex items-start justify-between border-b border-border-default px-6 py-5">
          <div class="space-y-1">
            <DialogTitle class="text-base font-semibold text-fg">
              <slot name="title">Details</slot>
            </DialogTitle>
            <DialogDescription class="text-xs text-fg-muted">
              <slot name="description" />
            </DialogDescription>
          </div>
          <DialogClose
            aria-label="Close"
            class="-mr-2 -mt-1 inline-flex h-8 w-8 items-center justify-center rounded-md
                   text-fg-muted hover:bg-surface-muted hover:text-fg
                   focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/40"
          >
            <X :size="16" />
          </DialogClose>
        </div>

        <div class="min-h-0 flex-1 overflow-hidden">
          <slot />
        </div>

        <div
          v-if="$slots.footer"
          class="border-t border-border-default px-6 py-4"
        >
          <slot name="footer" />
        </div>
      </DialogContent>
    </DialogPortal>
  </DialogRoot>
</template>

<script setup>
import {
  DialogRoot,
  DialogPortal,
  DialogOverlay,
  DialogContent,
  DialogTitle,
  DialogDescription,
  DialogClose,
} from 'reka-ui';
import { X } from 'lucide-vue-next';
import { cn } from '@/lib/cn';

defineProps({
  open: { type: Boolean, required: true },
  /** @type {'right' | 'left'} */
  side: { type: String, default: 'right' },
});
defineEmits(['update:open']);

const sideClasses = {
  right:
    'inset-y-0 right-0 h-full w-full max-w-[720px] border-l border-border-default ' +
    'data-[state=open]:animate-in data-[state=open]:slide-in-from-right ' +
    'data-[state=closed]:animate-out data-[state=closed]:slide-out-to-right',
  left:
    'inset-y-0 left-0 h-full w-full max-w-[720px] border-r border-border-default ' +
    'data-[state=open]:animate-in data-[state=open]:slide-in-from-left ' +
    'data-[state=closed]:animate-out data-[state=closed]:slide-out-to-left',
};
</script>
