import axiosInstance from "@/stores/axios-api";
import { MoodResponse, Mood } from "@/types/moods";

const moodsEndpoint = '/api/journals/moods/';

export type MoodParams = {
  name?: string | null;
  score_max?: number;
  score_min?: number;
  page: number;
}

export const getMoods = async (params: MoodParams) => {
  const response = await axiosInstance.get<MoodResponse>(moodsEndpoint, { params });
  return response.data;
};

export const getMoodById = async (id: string) => {
  const response = await axiosInstance.get<Mood>(`${moodsEndpoint}${id}/`);
  return response.data;
};

export const createMood = async (data: Partial<Mood>) => {
  const response = await axiosInstance.post<Mood>(moodsEndpoint, data);
  return response.data;
};

export const updateMood = async (id: string, data: Partial<Mood>) => {
  const response = await axiosInstance.put<Mood>(`${moodsEndpoint}${id}/`, data);
  return response.data;
};

export const patchMood = async (id: string, data: Partial<Mood>) => {
  const response = await axiosInstance.patch<Mood>(`${moodsEndpoint}${id}/`, data);
  return response.data;
};

export const deleteMood = async (id: string) => {
  await axiosInstance.delete(`${moodsEndpoint}${id}/`);
};