import axios from 'axios';
import type {Audit,Case,Cohort,Component,ForecastResponse,History,Log,Metrics,ModelInfo,Quantity,Status} from './types';
export const http=axios.create({baseURL:'/api/v1',timeout:120000});
export const api={
 status:async()=> (await http.get<Status>('/status')).data,
 forecast:async(origin:string,kind:Case='normal')=>(await http.post<ForecastResponse>('/forecast',{origin:origin.replace('T',' '),case:kind,demo:kind==='blocked',coverage:'IN-national'})).data,
 history:async(start:string,end:string,offset=0,limit=5000)=>(await http.get<History>('/history',{params:{start,end,offset,limit}})).data,
 metrics:async(quantity:Quantity,horizon:number,cohort:Cohort,month:string)=>(await http.get<Metrics>('/metrics',{params:{quantity,horizon,cohort,month}})).data,
 audit:async(horizon:number,cohort:Cohort,month:string)=>(await http.get<Audit>('/audit',{params:{horizon,cohort,month}})).data,
 model:async(component:Component,horizon:number)=>(await http.get<ModelInfo>('/models',{params:{component,horizon}})).data,
 logs:async()=>(await http.get<{rows:Log[];context:string}>('/logs')).data,
};
export function errorText(error:unknown):string {return axios.isAxiosError(error)?String(error.response?.data?.error??error.message):String(error)}
