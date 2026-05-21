<!--
  Tabs — wraps Reka UI TabsRoot.

  Usage:
    <Tabs v-model="activeTab" :items="[
      { value: 'overview', label: 'Overview' },
      { value: 'logs', label: 'Logs' },
    ]">
      <template #overview>...</template>
      <template #logs>...</template>
    </Tabs>
-->
<template>
  <TabsRoot
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    class="flex h-full flex-col"
  >
    <TabsList
      class="flex shrink-0 items-center gap-1 border-b border-border-default px-6"
    >
      <TabsTrigger
        v-for="item in items"
        :key="item.value"
        :value="item.value"
        :class="cn(
          'relative -mb-px inline-flex h-10 items-center border-b-2 border-transparent',
          'px-3 text-sm font-medium text-fg-muted',
          'hover:text-fg',
          'data-[state=active]:border-brand data-[state=active]:text-fg',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/40 focus-visible:ring-inset',
        )"
      >
        {{ item.label }}
      </TabsTrigger>
    </TabsList>

    <TabsContent
      v-for="item in items"
      :key="item.value"
      :value="item.value"
      class="min-h-0 flex-1 overflow-auto px-6 py-5 focus:outline-none"
    >
      <slot :name="item.value" />
    </TabsContent>
  </TabsRoot>
</template>

<script setup>
import { TabsRoot, TabsList, TabsTrigger, TabsContent } from 'reka-ui';
import { cn } from '@/lib/cn';

defineProps({
  modelValue: { type: String, required: true },
  /** @type {Array<{value: string, label: string}>} */
  items: { type: Array, required: true },
});
defineEmits(['update:modelValue']);
</script>
