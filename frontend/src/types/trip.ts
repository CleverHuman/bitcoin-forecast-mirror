import { BasePaginationParams } from "@/hooks/use-api-pagination";
import { BaseResponse } from "./base-response"
import { Journal } from "./journal"

export interface GetTrips extends BasePaginationParams {
  title?: string
}

export interface TripData {
  id?: string,
  title?: string,
  description?: string,
  start_date?: string | null;
  end_date?: string | null;
  journal_entries?: Journal[];
  created_at?: string
}

export interface TripListResponse extends BaseResponse<TripData[]> {}

export interface TripDetailResponse extends BaseResponse<TripData> {}


