import type {EChartsCoreOption} from 'echarts/core';
import type {Row} from './types';
export const palette={demand:'#00D9FF',wind:'#287BFF',solar:'#FFB84D',net:'#956BFF',renewable:'#24D69A'};
export function lines(rows:Row[],series:{key:string;name:string;color:string;dashed?:boolean}[],unit='MW',zoom=false):EChartsCoreOption {
 const times=rows.map(r=>String(r.target_timestamp??r.timestamp));
 return {color:series.map(s=>s.color),backgroundColor:'transparent',textStyle:{color:'#8798af',fontFamily:'Inter'},grid:{left:62,right:20,top:24,bottom:zoom?70:48},
 tooltip:{trigger:'axis',backgroundColor:'#0B1425',borderColor:'#21344d',textStyle:{color:'#dce7f5'}},legend:{bottom:0,textStyle:{color:'#91a4bc'},itemWidth:14,itemHeight:2},
 xAxis:{type:'category',data:times,boundaryGap:false,axisLabel:{color:'#73869f',fontSize:10,formatter:(v:string)=>v.replace('T',' ').slice(5,16)},axisLine:{lineStyle:{color:'#203047'}},axisTick:{show:false}},
 yAxis:{type:'value',name:unit,nameTextStyle:{color:'#73869f'},axisLabel:{color:'#73869f',fontSize:10,formatter:(v:number)=>Math.abs(v)>=10000?`${Math.round(v/1000)}k`:String(v)},splitLine:{lineStyle:{color:'#18253a',type:'dashed'}}},
 ...(zoom?{dataZoom:[{type:'inside'},{type:'slider',height:15,bottom:28,borderColor:'#203047',fillerColor:'#00d9ff15',textStyle:{color:'#73869f'}}]}:{}),
 series:series.map(s=>({name:s.name,type:'line',data:rows.map(r=>r[s.key]??null),connectNulls:false,showSymbol:rows.length<=30,symbolSize:4,sampling:'lttb',lineStyle:{width:2,type:s.dashed?'dashed':'solid'},...(s.dashed?{}:{areaStyle:{opacity:.045}})}))};
}
