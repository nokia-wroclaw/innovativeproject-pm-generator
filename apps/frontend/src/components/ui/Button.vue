<template>
  <button
    :type="type"
    :disabled="disabled"
    :class="cn(buttonVariants({ variant, size }), $attrs.class)"
    v-bind="rest"
  >
    <slot />
  </button>
</template>

<script setup>
import { computed, useAttrs } from 'vue';
import { cva } from 'class-variance-authority';
import { cn } from '@/lib/cn';

defineOptions({ inheritAttrs: false });

const props = defineProps({
  variant: { type: String, default: 'primary' },
  size: { type: String, default: 'md' },
  type: { type: String, default: 'button' },
  disabled: { type: Boolean, default: false },
});

const buttonVariants = cva(
  [
    'inline-flex items-center justify-center gap-2 rounded-md font-medium',
    'transition-colors duration-150',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/40 focus-visible:ring-offset-2',
    'disabled:cursor-not-allowed disabled:opacity-50',
  ].join(' '),
  {
    variants: {
      variant: {
        primary: 'bg-brand text-white hover:bg-brand-strong',
        secondary: 'bg-surface text-fg border border-border-default hover:bg-surface-muted',
        ghost: 'bg-transparent text-fg-muted hover:bg-surface-muted',
        danger: 'bg-rose-500 text-white hover:bg-rose-600',
      },
      size: {
        sm: 'h-8 px-3 text-xs',
        md: 'h-10 px-4 text-sm',
        icon: 'h-9 w-9 p-0',
      },
    },
    defaultVariants: { variant: 'primary', size: 'md' },
  },
);

const attrs = useAttrs();
const rest = computed(() => {
  const { class: _omit, ...rest } = attrs;
  return rest;
});

void props;
</script>
