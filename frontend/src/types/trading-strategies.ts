import { BaseResponse } from './base-response'

export interface Rule {
  id?: number | null
  condition: string
  action: string
  key?: number | null
}

export interface Parameter {
  id?: number | null
  name: string
  value: string
  key?: number | null
}

export interface Strategy {
  id: number | null
  name: string
  description?: string
  is_global: boolean
  rules: Rule[]
  parameters: Parameter[]
}

export type StrategiesResponse = BaseResponse<Strategy[]>

export interface ValidationErrors {
  rules?: { condition?: string[]; action?: string[] }[];
  parameters?: { name?: string[]; value?: string[] }[];
}
