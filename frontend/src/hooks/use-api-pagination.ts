import { useState, useCallback, useEffect } from 'react'
import { BaseResponse } from '@/types/base-response'

interface UseApiPaginationOptions<T, P> {
  fetchFn: (params: P) => Promise<BaseResponse<T[]>>
  pageSize?: number
  initialParams?: P
}

interface PaginationState<T> {
  data: T[]
  isLoading: boolean
  pageCount: number
  currentPage: number
  isInitialized: boolean
  hasNextPage: boolean 
}

export interface BasePaginationParams {
  page: number,
  page_size?: number
}

export function useApiPagination<T, P extends BasePaginationParams>({
  fetchFn,
  pageSize = 10,
  initialParams
}: UseApiPaginationOptions<T, P>) {
  const [state, setState] = useState<PaginationState<T>>({
    data: [],
    isLoading: false,
    currentPage: 1,
    isInitialized: false,
    pageCount: 1,
    hasNextPage: true 
  })

  const fetchData = useCallback(
    async (page: number, params: Partial<P> = initialParams ?? {}) => {
      if (state.isLoading) return
      if (page > state.currentPage && !state.hasNextPage) return  

      setState(prev => ({ ...prev, isLoading: true }))
      try {
        const response = await fetchFn({
          ...params,
          page,
          page_size: pageSize,
        } as P)

        setState({
          data: response.results,
          pageCount: response.count ? Math.ceil(response.count / pageSize) : 1,
          isLoading: false,
          currentPage: page,
          isInitialized: true,
          hasNextPage: response.next !== null 
        })
      } catch (error) {
        setState(prev => ({ 
          ...prev, 
          isLoading: false,
          isInitialized: true
        }))
      }
    },
    [fetchFn, pageSize, state.isLoading]
  )

  const updateParams = useCallback((newParams: Partial<P>) => {
    fetchData(1, newParams)
  }, [fetchData])

  useEffect(() => {
    fetchData(1)
  }, [])

  const goToPage = useCallback((page: number) => {
    if (!state.isLoading && page > 0 && page <= state.pageCount) {
      fetchData(page)
    }
  }, [fetchData, state.isLoading, state.pageCount])

  return {
    ...state,
    updateParams,
    goToPage,
    refresh: () => fetchData(1),
  }
}