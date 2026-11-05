import {describe,it,expect} from 'vitest';
import {clockKey,derived,overlayRows,shiftSourceClock} from './analytics';
import {csv,fmt} from './components/ui';
describe('scientific display transformations',()=>{
 it('preserves renewable surplus and shares above 100%',()=>{expect(derived(10,8,7)).toEqual({renewable:15,net:-5,share:150})});
 it('keeps undefined ratios and missing/nonfinite components missing',()=>{expect(derived(0,8,7)).toEqual({renewable:15,net:-15,share:null});expect(derived(10,null,7)).toEqual({renewable:null,net:null,share:null});expect(derived(10,Number.NaN,7).net).toBeNull();expect(fmt(Number.NaN)).toBe('—')});
 it('does source-clock arithmetic without browser timezone shifts',()=>{expect(shiftSourceClock('2025-06-24 23:00:00',24)).toBe('2025-06-25 23:00:00');expect(shiftSourceClock('2024-02-28T23:00:00',1)).toBe('2024-02-29 00:00:00');expect(clockKey('2025-06-25T23:00:00.000')).toBe('2025-06-25 23:00:00')});
 it('attaches actuals after forecasting and does not change target quality',()=>{const input=[{target_timestamp:'2025-01-01T01:00:00',demand_mw:10,target_quality:'unknown'}];const copy=JSON.stringify(input);const result=overlayRows(input,[{timestamp:'2025-01-01 01:00:00',demand_mw:11,wind_mw:2,solar_mw:3}]);expect(result[0].actual_demand_mw).toBe(11);expect(result[0].target_quality).toBe('unknown');expect(JSON.stringify(input)).toBe(copy);expect(overlayRows(input,[])[0].actual_demand_mw).toBeNull()});
 it('exports nulls as missing and escapes CSV cells without zero substitution',()=>{expect(csv([{value:null,method:'baseline,"quoted"'}])).toContain(',"baseline,""quoted"""');expect(csv([])).toBe('')});
});
