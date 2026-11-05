import {createContext,useContext} from 'react';
import type {Case} from './types';
interface ForecastActions {pending:boolean;error:string|null;run:(origin?:string,kind?:Case)=>void}
export const ForecastContext=createContext<ForecastActions>({pending:false,error:null,run:()=>{}});
export const useForecastActions=()=>useContext(ForecastContext);
