import axiosInstance from '@/stores/axios-api';

export type ExampleResponse = {
  id: number;
  name: string;
  description: string;
};

const exampleEndpoint = '/example';

export const getExample = async (): Promise<ExampleResponse> => {
  const response = await axiosInstance.get<ExampleResponse>(exampleEndpoint);
  return response.data;
};

