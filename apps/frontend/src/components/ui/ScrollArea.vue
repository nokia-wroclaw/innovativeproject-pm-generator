<!--
  ScrollArea — custom-styled scroll container.

  Wraps Reka UI's ScrollAreaRoot. We use this for log viewers and any other
  surface where we want a thin, neutral scrollbar rather than the platform
  default.
-->
<template>
  <ScrollAreaRoot
    :class="cn('relative overflow-hidden', $attrs.class)"
    :scroll-hide-delay="300"
  >
    <ScrollAreaViewport class="h-full w-full">
      <slot />
    </ScrollAreaViewport>

    <ScrollAreaScrollbar
      orientation="vertical"
      class="flex w-2 touch-none select-none rounded-full bg-transparent p-0.5"
    >
      <ScrollAreaThumb class="relative flex-1 rounded-full bg-slate-300" />
    </ScrollAreaScrollbar>

    <ScrollAreaScrollbar
      orientation="horizontal"
      class="flex h-2 touch-none select-none rounded-full bg-transparent p-0.5"
    >
      <ScrollAreaThumb class="relative flex-1 rounded-full bg-slate-300" />
    </ScrollAreaScrollbar>

    <ScrollAreaCorner class="bg-slate-100" />
  </ScrollAreaRoot>
</template>

<script setup>
import {
  ScrollAreaRoot,
  ScrollAreaViewport,
  ScrollAreaScrollbar,
  ScrollAreaThumb,
  ScrollAreaCorner,
} from 'reka-ui';
import { cn } from '@/lib/cn';

defineOptions({ inheritAttrs: false });
</script>
