import {useEffect,useRef} from 'react';
import * as echarts from 'echarts/core';
import {LineChart,BarChart,ScatterChart} from 'echarts/charts';
import {GridComponent,TooltipComponent,LegendComponent,DataZoomComponent,MarkLineComponent} from 'echarts/components';
import {CanvasRenderer} from 'echarts/renderers';
import type {EChartsCoreOption} from 'echarts/core';
echarts.use([LineChart,BarChart,ScatterChart,GridComponent,TooltipComponent,LegendComponent,DataZoomComponent,MarkLineComponent,CanvasRenderer]);
export default function Chart({option,height=260,group}:{option:EChartsCoreOption;height?:number;group?:string}) {
 const ref=useRef<HTMLDivElement>(null);
 const instance=useRef<ReturnType<typeof echarts.init>|null>(null);
 useEffect(()=>{if(!ref.current)return;const chart=echarts.init(ref.current,undefined,{renderer:'canvas'});instance.current=chart;if(group){chart.group=group;echarts.connect(group)}
 const observer=new ResizeObserver(()=>chart.resize());observer.observe(ref.current);
 const visibility=()=>chart.setOption({animation:!document.hidden&&!window.matchMedia('(prefers-reduced-motion: reduce)').matches});document.addEventListener('visibilitychange',visibility);
 return()=>{observer.disconnect();document.removeEventListener('visibilitychange',visibility);chart.dispose();instance.current=null};},[group]);
 useEffect(()=>{instance.current?.setOption({...option,animation:!window.matchMedia('(prefers-reduced-motion: reduce)').matches&&!document.hidden},true)},[option]);
 return <div ref={ref} style={{height,width:'100%'}} className="echart" role="img" aria-label="Interactive analytical chart"/>;
}
