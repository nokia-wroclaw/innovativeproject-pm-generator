import { computed, unref } from 'vue';
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';

import * as Api from '../services/modelingApi.js';

const THREE_SECONDS = 3_000;
const FIFTEEN_SECONDS = 15_000;

const ONE_HOUR = 60 * 60 * 1000;

export const modelingQueryKeys = {
  datasets: () => ['modeling', 'datasets'],
  models: () => ['modeling', 'models'],
  formSchema: (processType) => ['modeling', 'form-schema', unref(processType)],
  runStatus: (processType, runId) => [
    'modeling',
    'processes',
    unref(processType),
    'runs',
    unref(runId),
  ],
};

export function useModelingDatasets() {
  return useQuery({
    queryKey: modelingQueryKeys.datasets(),
    queryFn: Api.listModelingDatasets,
    refetchInterval: FIFTEEN_SECONDS,
  });
}

export function useModelingModels() {
  return useQuery({
    queryKey: modelingQueryKeys.models(),
    queryFn: Api.listModelingModels,
    staleTime: ONE_HOUR,
  });
}

export function useModelingFormSchema(processTypeRef) {
  return useQuery({
    queryKey: computed(() => modelingQueryKeys.formSchema(processTypeRef)),
    queryFn: ({ signal }) => Api.getModelingFormSchema(unref(processTypeRef), { signal }),
    enabled: computed(() => Boolean(unref(processTypeRef))),
    retry: 0,
    staleTime: Number.POSITIVE_INFINITY,
  });
}

export function useTriggerModelingRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ processType, body }) => Api.triggerModelingRun(processType, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: modelingQueryKeys.datasets() });
    },
  });
}

export function useModelingRunStatus(processTypeRef, runIdRef) {
  return useQuery({
    queryKey: computed(() => modelingQueryKeys.runStatus(processTypeRef, runIdRef)),
    queryFn: () => Api.getModelingRunStatus(unref(processTypeRef), unref(runIdRef)),
    enabled: computed(() => Boolean(unref(processTypeRef) && unref(runIdRef))),
    retry: 0,
    refetchInterval: THREE_SECONDS,
    refetchIntervalInBackground: false,
  });
}
