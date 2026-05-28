<!--
  TriggerDagDialog — modal for triggering a DAG with optional `conf` JSON.

  Validates JSON on input and on submit; disables the action button when the
  payload is invalid. Format button pretty-prints the value.

  Emits:
    - `update:open`     two-way binding for visibility
    - `triggered`       fired after a successful trigger (parent invalidates queries)
-->
<template>
  <DialogRoot
    :open="open"
    @update:open="$emit('update:open', $event)"
  >
    <DialogPortal>
      <DialogOverlay
        class="fixed inset-0 z-40 bg-slate-900/40 backdrop-blur-[2px]
               data-[state=open]:animate-in data-[state=open]:fade-in-0
               data-[state=closed]:animate-out data-[state=closed]:fade-out-0"
      />
      <DialogContent
        class="fixed left-1/2 top-1/2 z-50 w-full max-w-lg -translate-x-1/2 -translate-y-1/2
               rounded-xl border border-border-default bg-surface shadow-xl
               data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95
               data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95"
      >
        <div class="flex items-start justify-between border-b border-border-default px-6 py-5">
          <div class="space-y-1">
            <DialogTitle class="text-base font-semibold text-fg">
              Trigger DAG
            </DialogTitle>
            <DialogDescription class="text-xs text-fg-muted">
              <span class="font-mono">{{ dagId }}</span> — uruchomi nowy run z opcjonalnym <code>conf</code> JSON.
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

        <div class="space-y-4 px-6 py-5">
          <div class="space-y-1.5">
            <label
              for="trigger-note"
              class="text-xs font-semibold uppercase tracking-wider text-fg-subtle"
            >
              Note (optional)
            </label>
            <input
              id="trigger-note"
              v-model="note"
              type="text"
              placeholder="e.g. backfill missing data for 2026-05-20"
              class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg
                     placeholder:text-fg-subtle
                     focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/40"
            />
          </div>

          <div class="space-y-1.5">
            <div class="flex items-center justify-between">
              <label
                for="trigger-conf"
                class="text-xs font-semibold uppercase tracking-wider text-fg-subtle"
              >
                Conf (JSON, optional)
              </label>
              <button
                type="button"
                :disabled="!confRaw.trim() || jsonError !== null"
                @click="formatJson"
                class="text-[11px] font-medium text-brand hover:text-brand-strong disabled:cursor-not-allowed disabled:opacity-50"
              >
                Format
              </button>
            </div>
            <textarea
              id="trigger-conf"
              v-model="confRaw"
              rows="8"
              spellcheck="false"
              placeholder='{ "param": "value" }'
              :class="cn(
                'w-full rounded-md border bg-surface px-3 py-2 font-mono text-xs text-fg',
                'placeholder:text-fg-subtle',
                'focus-visible:outline-none focus-visible:ring-2',
                jsonError
                  ? 'border-rose-300 focus-visible:ring-rose-300'
                  : 'border-border-default focus-visible:ring-brand/40',
              )"
            />
            <p
              v-if="jsonError"
              class="flex items-start gap-1.5 text-[11px] text-rose-600"
            >
              <AlertCircle :size="12" class="mt-0.5 shrink-0" />
              {{ jsonError }}
            </p>
            <p v-else-if="confRaw.trim()" class="text-[11px] text-emerald-600">
              JSON valid.
            </p>
            <p v-else class="text-[11px] text-fg-subtle">
              Leave blank to trigger with the DAG's default configuration.
            </p>
          </div>

          <div
            v-if="serverError"
            class="rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700"
          >
            <p class="font-semibold">Trigger failed</p>
            <p class="mt-1">{{ serverError }}</p>
          </div>
        </div>

        <div class="flex items-center justify-end gap-2 border-t border-border-default px-6 py-4">
          <Button variant="secondary" size="sm" :disabled="isSubmitting" @click="onCancel">
            Cancel
          </Button>
          <Button
            variant="primary"
            size="sm"
            :disabled="jsonError !== null || isSubmitting"
            @click="onSubmit"
          >
            <Play :size="14" />
            {{ isSubmitting ? 'Triggering…' : 'Trigger' }}
          </Button>
        </div>
      </DialogContent>
    </DialogPortal>
  </DialogRoot>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import {
  DialogRoot, DialogPortal, DialogOverlay, DialogContent,
  DialogTitle, DialogDescription, DialogClose,
} from 'reka-ui';
import { Play, X, AlertCircle } from 'lucide-vue-next';
import { cn } from '@/lib/cn';
import { Button } from '@/components/ui';
import { useTriggerDag } from '../composables/queries.js';

const props = defineProps({
  open: { type: Boolean, required: true },
  dagId: { type: String, required: true },
});
const emit = defineEmits(['update:open', 'triggered']);

const note = ref('');
const confRaw = ref('');
const serverError = ref('');

const triggerMutation = useTriggerDag();
const isSubmitting = computed(() => triggerMutation.isPending.value);

const jsonError = computed(() => {
  const raw = confRaw.value.trim();
  if (!raw) return null;
  try {
    JSON.parse(raw);
    return null;
  } catch (e) {
    return e.message;
  }
});

function formatJson() {
  try {
    confRaw.value = JSON.stringify(JSON.parse(confRaw.value), null, 2);
  } catch {
    /* button disabled when invalid, defensive only */
  }
}

function onCancel() {
  emit('update:open', false);
}

async function onSubmit() {
  if (jsonError.value) return;
  serverError.value = '';
  const body = {};
  const raw = confRaw.value.trim();
  if (raw) body.conf = JSON.parse(raw);
  if (note.value.trim()) body.note = note.value.trim();

  try {
    const result = await triggerMutation.mutateAsync({ dagId: props.dagId, body });
    emit('triggered', result);
    emit('update:open', false);
  } catch (err) {
    serverError.value = err?.message ?? 'Trigger failed';
  }
}

// Reset form when the dialog closes.
watch(
  () => props.open,
  (open) => {
    if (!open) {
      note.value = '';
      confRaw.value = '';
      serverError.value = '';
    }
  },
);
</script>
