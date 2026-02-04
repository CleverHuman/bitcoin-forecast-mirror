import axiosInstance from "@/stores/axios-api";
import { JournalResponse, Journal } from "@/types/journal";


const journalsEndpoint = '/api/journals/';

export type JournalParams = {
  background_color?: string | string[];
  journal_date?: string;
  mood_id?: string | string[];
  page: number;
  title?: string;
  page_size?: number | 10;
}

export const getJournals = async (params: JournalParams) => {
  const formattedParams = { ...params };
  
  if (Array.isArray(formattedParams.mood_id)) {
    formattedParams.mood_id = JSON.stringify(formattedParams.mood_id);
  }
  if (Array.isArray(formattedParams.background_color)) {
    formattedParams.background_color = JSON.stringify(formattedParams.background_color);
  }

  Object.keys(formattedParams).forEach(key => {
    if (formattedParams[key as keyof typeof formattedParams] === undefined) {
      delete formattedParams[key as keyof typeof formattedParams];
    }
  });

  const response = await axiosInstance.get<JournalResponse>(journalsEndpoint, { 
    params: formattedParams
  });
  return response.data;
};

export const getJournalById = async (id: string) => {
  const response = await axiosInstance.get<Journal>(`${journalsEndpoint}${id}/`);
  return response.data;
};

export const createJournal = async (data: Partial<Journal>) => {
  const response = await axiosInstance.post<Journal>(journalsEndpoint, data);
  return response.data;
};

export const updateJournal = async (id: string, data: Partial<Journal>) => {
  const response = await axiosInstance.put<Journal>(`${journalsEndpoint}${id}/`, data);
  return response.data;
};

export const patchJournal = async (id: string, data: Partial<Journal>) => {
  const response = await axiosInstance.patch<Journal>(`${journalsEndpoint}${id}/`, data);
  return response.data;
};

export const deleteJournal = async (id: string) => {
  await axiosInstance.delete(`${journalsEndpoint}${id}/`);
};
