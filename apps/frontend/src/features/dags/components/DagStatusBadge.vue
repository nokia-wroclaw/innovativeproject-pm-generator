<template>
  <span :class="cn(badgeVariants({ status, density }), $attrs.class)">
    <component
      :is="icon"
      :size="density === 'compact' ? 12 : 14"
      :class="cn('shrink-0', spinIcon && 'animate-spin')"
      aria-hidden="true"
    />
    <span v-if="density !== 'icon-only'">{{ label }}</span>
  </span>
</template>

<script setup>
import { computed } from 'vue';
import { cva } from 'class-variance-authority';
import { CheckCircle2, Loader2, XCircle, Clock } from 'lucide-vue-next';
import { cn } from '@/lib/cn';

defineOptions({ inheritAttrs: false });

const props = defineProps({
  status: { type: String, required: true },
  density: { type: String, default: 'default' },
});

const STATUS_META = {
  success: { icon: CheckCircle2, label: 'Success' },
  running: { icon: Loader2, label: 'Running' },
  failed: { icon: XCircle, label: 'Failed' },
  queued: { icon: Clock, label: 'Queued' },
};

const icon = computed(() => STATUS_META[props.status]?.icon ?? Clock);
const label = computed(() => STATUS_META[props.status]?.label ?? 'Unknown');
const spinIcon = computed(() => props.status === 'running');

const badgeVariants = cva(
  'inline-flex items-center gap-1.5 font-medium ring-1 ring-inset',
  {
    variants: {
      status: {
        success: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
        running: 'bg-sky-50 text-sky-700 ring-sky-200',
        failed: 'bg-rose-50 text-rose-700 ring-rose-200',
        queued: 'bg-violet-50 text-violet-700 ring-violet-200',
      },
      density: {
        default: 'rounded-full px-2.5 py-1 text-xs',
        compact: 'rounded-md px-1.5 py-0.5 text-[11px]',
        'icon-only': 'rounded-full h-6 w-6 justify-center p-0',
      },
    },
    defaultVariants: { status: 'queued', density: 'default' },
  },
);
</script>
