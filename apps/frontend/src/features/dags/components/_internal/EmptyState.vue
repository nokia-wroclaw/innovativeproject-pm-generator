<!--
  EmptyState — friendly placeholder block for tabs/panels with no data yet.
  Internal to features/dags.
-->
<template>
  <div class="flex h-full flex-col items-center justify-center gap-3 py-12 text-center">
    <component
      :is="iconComponent"
      :size="28"
      class="text-fg-subtle"
      aria-hidden="true"
    />
    <div class="space-y-1 max-w-xs">
      <p v-if="title" class="text-sm font-semibold text-fg">
        {{ title }}
      </p>
      <p class="text-xs text-fg-muted leading-relaxed">
        {{ message }}
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { Inbox, Terminal, Boxes } from 'lucide-vue-next';

const ICONS = { inbox: Inbox, terminal: Terminal, boxes: Boxes };

const props = defineProps({
  /** @type {'inbox' | 'terminal' | 'boxes'} */
  iconName: { type: String, default: 'inbox' },
  title: { type: String, default: '' },
  message: { type: String, required: true },
});

const iconComponent = computed(() => ICONS[props.iconName] ?? Inbox);
</script>
