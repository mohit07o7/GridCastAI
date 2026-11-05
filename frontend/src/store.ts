import {create} from 'zustand';
import type {Cohort,ForecastResponse} from './types';
interface State {draftOrigin:string;horizon:number;forecast:ForecastResponse|null;cohort:Cohort;month:string;overlay:boolean;setDraft:(v:string)=>void;setHorizon:(v:number)=>void;setForecast:(v:ForecastResponse)=>void;setCohort:(v:Cohort)=>void;setMonth:(v:string)=>void;setOverlay:(v:boolean)=>void}
export const useDesk=create<State>((set)=>({draftOrigin:'2025-06-24T23:00',horizon:24,forecast:null,cohort:'delivered',month:'all',overlay:false,
 setDraft:v=>set({draftOrigin:v}),setHorizon:v=>set({horizon:v}),setForecast:v=>set({forecast:v}),setCohort:v=>set({cohort:v}),setMonth:v=>set({month:v}),setOverlay:v=>set({overlay:v})}));
