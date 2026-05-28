<!--
  DefRow — a single "label / value" row used inside the details sheet.
  Internal to features/dags; not part of the public component surface.
-->
<template>
  <div class="min-w-0">
    <dt class="text-[11px] font-semibold uppercase tracking-wider text-fg-subtle">
      {{ label }}
    </dt>
    <dd class="mt-0.5 truncate font-mono text-sm text-fg">
      <slot />
    </dd>
  </div>
</template>

<script setup>
defineProps({
  label: { type: String, required: true },
});
</script>
