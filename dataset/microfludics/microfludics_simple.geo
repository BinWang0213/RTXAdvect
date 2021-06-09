SetFactory("OpenCASCADE");

hsub = 0.5; //Level 1 = 0.5, Level 4 = 4.5, Level 5 = 5, 
scale = 1;

hmax=0.4/hsub;
hmin=hmax/10;

Mesh.CharacteristicLengthMax = hmax;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 25;


Merge "CAD_2D_simple.stp";

vol_thick[] = 
Extrude { 0.0,-0.4,0.0 }
{
	Surface{:};
	Layers{8};
	//Recombine;
};

Coherence;

//Circle
Field[1] = Distance;
Field[1].NNodesByEdge = 100;
Field[1].FacesList = {1,76, 59,60, 67,5, 17,71, 9, 63, 41,42
                     };

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = hmin*5;
Field[2].LcMax = hmax;
Field[2].DistMin = 0.5;
Field[2].DistMax = 1.0;

//Small pillar feature
Field[3] = Distance;
Field[3].NNodesByEdge = 100;
Field[3].FacesList = {75,76,
					  68,67,
					  63,64,
					  71,72
                      };

Field[4] = Threshold;
Field[4].IField = 3;
Field[4].LcMin = hmin*2;
Field[4].LcMax = hmax;
Field[4].DistMin = 0.22;
Field[4].DistMax = 1.0;

Field[7] = Min;
Field[7].FieldsList = {2,4};
Background Field = 7;

Physical Surface("inlet") = {13};
Physical Surface("outlet") = {21};
Physical Volume("domain") = {1:5};

Mesh.ScalingFactor = 100;


