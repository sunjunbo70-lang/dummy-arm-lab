// DRAFT ONLY: additive extension; fuse root with existing trowel holder.
// Coordinate: X along handle, Y outward from flange, Z sideways.
// D435 rear M3, 45 mm centers, max screw insertion 3 mm.
// Not a complete flange/clamp CAD. Validate actual D435f and cable clearance.
$fn=64;
z=70.0; t=6.0; pitch=45.0; bore=3.2;
difference() {
 union() {
  translate([-8,0,20]) cube([16,t,z-20]);
  translate([-4,-6,20]) cube([8,6,z-29]);
  translate([-pitch/2-6,0,z-9]) cube([pitch+12,t,18]);
 }
 for(x=[-pitch/2,pitch/2])
  translate([x,-1,z]) rotate([-90,0,0]) cylinder(h=t+2,d=bore);
}
