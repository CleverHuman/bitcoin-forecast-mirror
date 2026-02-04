import axiosInstance from "@/stores/axios-api";
import { GetTrips, TripData, TripDetailResponse, TripListResponse } from "@/types/trip";


const endpoint = '/api/journals/trips/';

export const getTrips = async (params: GetTrips) => {
  const formattedParams = { ...params };

  const response = await axiosInstance.get<TripListResponse>(endpoint, {
    params: formattedParams
  });
  return response.data;
};

export const getTripById = async (id: string) => {
  const response = await axiosInstance.get<TripDetailResponse>(`${endpoint}${id}/`);
  return response.data;
};

export const createTrip = async (data: Partial<TripData>) => {
  const response = await axiosInstance.post<TripDetailResponse>(endpoint, data);
  return response.data;
};

export const updateTrip = async (id: string, data: Partial<TripData>) => {
  const response = await axiosInstance.put<TripDetailResponse>(`${endpoint}${id}/`, data);
  return response.data;
};

export const patchTrip = async (id: string, data: Partial<TripData>) => {
  const response = await axiosInstance.patch<TripDetailResponse>(`${endpoint}${id}/`, data);
  return response.data;
};

export const deleteTrip = async (id: string) => {
  await axiosInstance.delete(`${endpoint}${id}/`);
};