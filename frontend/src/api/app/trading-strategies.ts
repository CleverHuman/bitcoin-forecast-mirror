import axiosInstance from "@/stores/axios-api";
import { StrategiesResponse, Strategy } from "@/types/trading-strategies";

const strategyEndpoint = '/api/trading/strategy/';

export type StrategyParams = {
  page?: number;
}

export const getStrategies = async (params: StrategyParams) => {
  const response = await axiosInstance.get<StrategiesResponse>(strategyEndpoint, { params });
  return response.data;
};

export const getStrategyById = async (id: string) => {
  const response = await axiosInstance.get<Strategy>(`${strategyEndpoint}${id}/`);
  return response.data;
};

export const createStrategy = async (data: Partial<Strategy>) => {
  const response = await axiosInstance.post<Strategy>(strategyEndpoint, data);
  return response.data;
};

export const updateStrategy = async (id: string, data: Partial<Strategy>) => {
  const response = await axiosInstance.put<Strategy>(`${strategyEndpoint}${id}/`, data);
  return response.data;
};

export const patchStrategy = async (id: string, data: Partial<Strategy>) => {
  const response = await axiosInstance.patch<Strategy>(`${strategyEndpoint}${id}/`, data);
  return response.data;
};

export const deleteStrategy = async (id: string) => {
  await axiosInstance.delete(`${strategyEndpoint}${id}/`);
};