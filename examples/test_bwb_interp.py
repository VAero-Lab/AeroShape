import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec, AirfoilProfile
import subprocess

def test_bwb():
    try:
        if os.path.exists("Exports/blended_wing_body_guided.step"):
            os.remove("Exports/blended_wing_body_guided.step")
        
        # We will dynamically overwrite the wire generation in AirfoilProfile just for this test
        original_to_occ = AirfoilProfile.to_occ_wire
        
        def patched_to_occ(self, z_off=0.0):
            from OCP.gp import gp_Pnt
            from OCP.GeomAPI import GeomAPI_Interpolate
            from OCP.TColgp import TColgp_HArray1OfPnt
            from OCP.TColStd import TColStd_HArray1OfReal
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
            
            px, pz = self.x.copy(), self.z.copy()
            px[-1], pz[-1] = px[0], pz[0]
            n = len(px)
            arr = TColgp_HArray1OfPnt(1, n)
            params = TColStd_HArray1OfReal(1, n)
            
            for i in range(n):
                arr.SetValue(i + 1, gp_Pnt(float(px[i]), 0.0, float(pz[i]) + z_off))
                params.SetValue(i + 1, i / (n - 1))
                
            interp = GeomAPI_Interpolate(arr, params, False, 1e-6)
            interp.Perform()
            if not interp.IsDone():
                raise RuntimeError("Failed to interpolate")
            edge = BRepBuilderAPI_MakeEdge(interp.Curve()).Edge()
            return BRepBuilderAPI_MakeWire(edge).Wire()
            
        AirfoilProfile.to_occ_wire = patched_to_occ
        
        # Run BWB guided export logic directly here
        import examples.aircraft_bwb_guided as bwb
        print("Running BWB Guided test with Interpolation...")
        t0 = time.time()
        # Create Wing
        wing = bwb.create_bwb_wing()
        shape = wing.to_occ_shape()
        
        from aeroshape.nurbs.export import NurbsExporter
        NurbsExporter.to_step(shape, "Exports/blended_wing_body_guided.step")
        t1 = time.time()
        
        sz = os.path.getsize("Exports/blended_wing_body_guided.step")
        print(f"BWB Guided size: {sz/1024/1024:.2f} MB, Time: {t1-t0:.2f} s")
        
        # Restore original
        AirfoilProfile.to_occ_wire = original_to_occ
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bwb()
