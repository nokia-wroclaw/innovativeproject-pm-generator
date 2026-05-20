<!--
  TaskStatusBadge — single source of truth for task-instance status visuals.

  Maps the 7 project-level statuses (TaskStatus) to color + icon + label.
  Any other component that needs to display a task status MUST go through
  this badge — never reach for `bg-emerald-500` directly. This keeps the
  design system honest (see contract §2.1).

  Two visual densities:
    - density="default"  → pill with icon + label (lists, headers)
    - density="compact"  → small chip, label only (table cells)
  Optional `pulse` prop animates the icon for live "running" states.
-->
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
import {
  CheckCircle2,
  Loader2,
  XCircle,
  RefreshCw,
  Clock,
  SkipForward,
  CircleDashed,
} from 'lucide-vue-next';
import { cn } from '@/lib/cn';

defineOptions({ inheritAttrs: false });

const props = defineProps({
  /**
   * @type {import('@/features/dags/types.js').TaskStatus}
   */
  status: { type: String, required: true },
  /** @type {'default' | 'compact' | 'icon-only'} */
  density: { type: String, default: 'default' },
});

const STATUS_META = {
  success: { icon: CheckCircle2, label: 'Success' },
  running: { icon: Loader2, label: 'Running' },
  failed: { icon: XCircle, label: 'Failed' },
  up_for_retry: { icon: RefreshCw, label: 'Retrying' },
  queued: { icon: Clock, label: 'Queued' },
  skipped: { icon: SkipForward, label: 'Skipped' },
  none: { icon: CircleDashed, label: 'No status' },
};

const icon = computed(() => STATUS_META[props.status]?.icon ?? CircleDashed);
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
        up_for_retry: 'bg-amber-50 text-amber-700 ring-amber-200',
        queued: 'bg-violet-50 text-violet-700 ring-violet-200',
        skipped: 'bg-slate-50 text-slate-600 ring-slate-200',
        none: 'bg-slate-50 text-slate-500 ring-slate-200',
      },
      density: {
        default: 'rounded-full px-2.5 py-1 text-xs',
        compact: 'rounded-md px-1.5 py-0.5 text-[11px]',
        'icon-only': 'rounded-full h-6 w-6 justify-center p-0',
      },
    },
    defaultVariants: { status: 'none', density: 'default' },
  },
);
</script>
