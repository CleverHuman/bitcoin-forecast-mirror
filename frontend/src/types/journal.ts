import { BaseResponse } from "./base-response"
import { Mood } from "./moods"
import { TripData } from "./trip"

export interface JournalEntry {
  id: string
  title: string
  body: string
  date: string
  mood: string
  pageColor: string
  previewImage?: string;
}

export type Journal = {
  id: string;
  user?: number;
  title?: string;
  notes?: string;
  background_color?: string;
  created_at?: string;
  updated_at?: string;
  mood_id?: string;
  mood?: Mood;
  tags?: Tag[];
  journal_date?: string;
  trip_id?: string
  trip?: TripData;
  isAddNewRow?: boolean;
}

export type Tag = {
  id: string;
  name: string;
}

export type JournalResponse = BaseResponse<Journal[]>


export interface JournalParams {
  page: number
  page_size: number
  title?: string
  mood_id?: string
  background_color?: string
  journal_date?: string
  trip_id?: string | undefined
}
