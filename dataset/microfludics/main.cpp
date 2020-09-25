// ======================================================================== //
// Copyright 2019-2020 The Collaborators                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //


#include "common.h"
#include "DeviceTetMesh.cuh"
#include "HostTetMesh.h"

#include "query/ConvexQuery.h"
#include "optix/OptixQuery.h"



//Switch between RTX and ConvexPoly
//#define ConvexPoly
#define RTX

namespace advect {

	//Input parameters
	int numParticles = 1e5;
	int testTetMeshGridSize = 45;
	//int numSteps = 550001;
	//int numSteps = 13251;
	int numSteps = 1e5;
	double dt = 5e-3;
	//double dt = 1e-2;
	//double dt = 1e-3;

	bool fixTimestep = true;

	double tol = 1e-5;
	double SeedingBox[6] = { 0.0 + tol, 0.0 + tol, 0.0 + tol,
							 1.0 - tol, 1.0 - tol, 1.0 - tol };

	//Physics controller
	bool usingAdvection = true;
	bool usingBrownianMotion = true;
	bool reflectWall = true;
	bool floatErrorCorrection = true;

	//IO container
	bool usingSeedingBox = false;
	bool saveStreamlinetoFile = false;
	int saveInterval = numSteps;
	//int saveInterval = 50000;

  extern "C" int main(int ac, char **av)
  {
    cudaTimer timer;
    timer.start();

	std::string vert_filename, tet_filename, velocity_vert_filename,velocity_tet_filename;
	std::string seeding_pts_filename;

	seeding_pts_filename = "SeedingPts.dat";
	seeding_pts_filename = "";
	vert_filename = "verts.dat";
	tet_filename = "cells.dat";
	velocity_tet_filename = "solutions_cell.dat";


	std::string objTrajectoryFileName;
	std::string vtkStreamlineFileName = "Streamline.vtk";
    for (int i=1;i<ac;i++) {
      const std::string arg = av[i];
      if (arg == "--num-particles")
        numParticles = std::atoi(av[++i]);
      else if (arg == "--num-steps")
        numSteps = std::atoi(av[++i]);
	  else if (arg == "--input_mesh") {
		  vert_filename = av[++i];
		  tet_filename = av[++i];
	  }
	  else if (arg == "--input_vertex_velocity_field") 
		  velocity_vert_filename= av[++i];
	  else if (arg == "--input_tet_velocity_field")
		  velocity_tet_filename = av[++i];
	  else if (arg == "--test-grid-size")
		  testTetMeshGridSize = std::atoi(av[++i]);
	  else if (arg == "-dt")
		  dt = std::atof(av[++i]);
	  else if (arg == "--seeding-box") {
		  for (int si = 0; si < 6; ++si)
			  SeedingBox[si] = std::atof(av[++i]);
		  usingSeedingBox = true;
	  }
	  else if (arg == "--save-streamline-to-obj") {
		  objTrajectoryFileName = av[++i];
		  saveStreamlinetoFile = true;
	  }
	  else if (arg == "--save-streamline-to-vtk") {
		  vtkStreamlineFileName = av[++i];
		  saveStreamlinetoFile = true;
	  }
      else
        throw std::runtime_error("unknown cmdline argument '"+arg+"'");
    }

    // ------------------------------------------------------------------
    // create a host-side model
    // ------------------------------------------------------------------
	std::string VelocityInterpMethod = "VertexVelocity";
	HostTetMesh hostTetMesh;
	if (velocity_vert_filename.size() > 0) {
		hostTetMesh = HostTetMesh::readDataSet(vert_filename, tet_filename, velocity_vert_filename);
		VelocityInterpMethod = "VertexVelocity";
		printf("#adv: load vertex velocity field from file %s", velocity_vert_filename.c_str());
	}
	else if (velocity_tet_filename.size() > 0) {
		hostTetMesh = HostTetMesh::readDataSet(vert_filename, tet_filename, "", velocity_tet_filename);
		VelocityInterpMethod = "TetVelocity";
		printf("#adv: load tet velocity field from file %s", velocity_vert_filename.c_str());
	}
	else {
		hostTetMesh = HostTetMesh::createBoxMesh(testTetMeshGridSize, testTetMeshGridSize, testTetMeshGridSize);
		VelocityInterpMethod = "VertexVelocity";
		printf("#adv: using synthetic velocity field\n");
	}
	//Get the boundary mesh representation
	HostTetMesh hostBoundaryMesh = hostTetMesh.getBoundaryMesh();

    // ------------------------------------------------------------------
    // build the query accelerator first, before the cuda kernels
    // allocate their memory.
    // ------------------------------------------------------------------
    OptixQuery tetQueryAccelerator((double3 *) hostTetMesh.positions.data(),
                                      hostTetMesh.positions.size(),
                                      (int4 *) hostTetMesh.indices.data(),
                                      hostTetMesh.indices.size());

    // by now optix should have built all its data,and released
    // whatever temp memory it has used.
	OptixQuery triQueryAccelerator((double3*)hostBoundaryMesh.positions.data(),
									hostBoundaryMesh.positions.size(),
									(int4*)hostBoundaryMesh.indices.data(),
									hostBoundaryMesh.indices.size(), true);

    // ------------------------------------------------------------------
    // upload our own cuda data
    // ------------------------------------------------------------------
    DeviceTetMesh devMesh;
    devMesh.upload(hostTetMesh);

	DeviceBdMesh devBdMesh;
	devBdMesh.upload(hostBoundaryMesh);

    // ------------------------------------------------------------------
    // now run sample advection...
    // ------------------------------------------------------------------
	if(!fixTimestep)
	double dt=cudaEvalTimestep(hostTetMesh.indices.size(),
		devMesh.d_indices,
		devMesh.d_positions,
		devMesh.d_velocities,
		VelocityInterpMethod);

    // alloc particles and its properties
	
	// Cast simple double4 particles into OptixTetquery type
	Particle* d_particles = nullptr;
	if (seeding_pts_filename.size() > 0) 
		numParticles = loadNumParticles(seeding_pts_filename);
	cudaCheck(cudaMalloc(&d_particles, numParticles * sizeof(Particle)));
	int* d_particles_tetIDs = nullptr;
	cudaCheck(cudaMalloc(&d_particles_tetIDs, numParticles * sizeof(int)));

	int* d_particles_triIDs = nullptr;
	cudaCheck(cudaMalloc(&d_particles_triIDs, numParticles * sizeof(int)));

	int* d_particles_ConvextetIDs = nullptr;
	cudaCheck(cudaMalloc(&d_particles_ConvextetIDs, numParticles * sizeof(int)));
	cudaCheck(cudaMemset(d_particles_ConvextetIDs, -1, numParticles * sizeof(int)));

	vec4d* d_particle_vels =nullptr;
	cudaCheck(cudaMalloc(&d_particle_vels, numParticles * sizeof(vec4d)));
	cudaCheck(cudaMemset(d_particle_vels, 0.0, numParticles * sizeof(vec4d)));

	vec4d* d_particle_disps = nullptr;
	cudaCheck(cudaMalloc(&d_particle_disps, numParticles * sizeof(vec4d)));
	cudaCheck(cudaMemset(d_particle_disps, 0.0, numParticles * sizeof(vec4d)));

	curandState_t* rand_states = nullptr;;
	cudaCheck(cudaMalloc(&rand_states, numParticles * sizeof(curandState_t)));
	initRandomGenerator(numParticles, rand_states);

	// Create streamlines object
	std::vector<std::vector<vec3f>> trajectories;


	usingSeedingBox = true;
	//double seedBox[6] = { -0.05+tol,0.0+tol,2.5+tol, 0.05-tol,0.1-tol,2.55-tol };
	//double seedBox[6] = { -0.05 + tol,0.0 + tol,2.2165 + tol, 0.05 - tol,0.1 - tol,2.7835 - tol };
	double seedBox[6] = { 73.9 + tol,-0.4 + tol,-655.95 + tol, 77.7 - tol,0.0 - tol,-655.45 - tol };
	std::copy(seedBox, seedBox+6, SeedingBox);


    // initialize with random particles
	if (seeding_pts_filename.size() == 0) {
		box3d initBox;
		if (usingSeedingBox) {
			initBox.extend(vec3d(SeedingBox[0], SeedingBox[1], SeedingBox[2]));
			initBox.extend(vec3d(SeedingBox[3], SeedingBox[4], SeedingBox[5]));
		}
		else {
			initBox = hostTetMesh.worldBounds;
		}
		//std::cout << "Particle seeding bounding box = " << initBox.lower << " " << initBox.upper << std::endl;
		printf("Particle Bounding Box (%lf,%lf,%lf)-(%lf,%lf,%lf)\n",
			initBox.lower.x, initBox.lower.y, initBox.lower.z,
			initBox.size().x, initBox.size().y, initBox.size().z);
		cudaInitParticles(d_particles, numParticles, initBox);
	}
	else 
		cudaInitParticles(d_particles, numParticles, seeding_pts_filename);

    cudaCheck(cudaDeviceSynchronize());
    
    printf("Init RunTime=%lf  ms\n", timer.stop());

	//Init initial state (pos,velocity,tetID)
	tetQueryAccelerator.query_sync(d_particles, d_particles_tetIDs, numParticles);
#ifdef  ConvexPoly
	tetQueryAccelerator.query_sync(d_particles, d_particles_ConvextetIDs, numParticles);
#endif

#ifndef  ConvexPoly
	cudaReportParticles(numParticles, d_particles_tetIDs);
#else
	cudaReportParticles(numParticles, d_particles_ConvextetIDs);
#endif

	if (usingAdvection) {
	
	cudaAdvect(d_particles,
#ifndef  ConvexPoly
		d_particles_tetIDs,
#else
		d_particles_ConvextetIDs,
#endif
		d_particle_vels,
		d_particle_disps,
		dt,
		numParticles,
		devMesh.d_indices,
		devMesh.d_positions,
		devMesh.d_velocities,
		VelocityInterpMethod);
	/*
	cudaTubeAdvect(d_particles, d_particles_tetIDs,
		           d_particle_vels, d_particle_disps, dt, numParticles);
	*/
	}
	
	writeParticles2VTU(0, d_particles, d_particle_vels, d_particles_tetIDs, numParticles,
		d_particles_ConvextetIDs);

#ifndef  ConvexPoly
	testRTtracing(triQueryAccelerator,tetQueryAccelerator,devMesh,devBdMesh);
#else
	testNStracing(tetQueryAccelerator, devMesh);
#endif

	system("pause");

	//VelocityInterpMethod = "ConstantVelocity";
	//VelocityInterpMethod = "VertexVelocity";

	// and iterate
    timer.start();
    for (int i=1;i<=numSteps;i++) {
		printf("------------Step %d-------------\n",i);
		// first, compute each particle's current tet for velocity interpolation
		//tetQueryAccelerator.query_sync(d_particles, d_particles_tetIDs, numParticles);

		// ... compute advection
		if (usingAdvection) {

			cudaAdvect(d_particles,
#ifndef  ConvexPoly
				d_particles_tetIDs,
#else
				d_particles_ConvextetIDs,
#endif
				d_particle_vels,
				d_particle_disps,
				dt,
				numParticles,
				devMesh.d_indices,
				devMesh.d_positions,
				devMesh.d_velocities,
				VelocityInterpMethod);
			/*
			cudaTubeAdvect(d_particles, d_particles_tetIDs,
						   d_particle_vels, d_particle_disps, dt, numParticles);
			*/
		}

		// ... compute random Brownian motion
		if(usingBrownianMotion)
		cudaBrownianMotion(d_particles, 
			d_particle_disps,
			rand_states,
			dt, 
			numParticles);

#ifndef  ConvexPoly
		// ... RTX Query next particle tet location
		//tetQueryAccelerator.query_sync(d_particles, (double4*) d_particle_disps, d_particles_tetIDs, numParticles);
		//tetQueryAccelerator.query_sync(d_particles, d_particles_tetIDs, numParticles);
#endif

#ifndef  ConvexPoly
		// ... compute wall reflection
		if(reflectWall)
		cudaWallReflection(triQueryAccelerator,
			d_particles,
			d_particles_triIDs,
			d_particle_vels,
			d_particle_disps,
			numParticles,
			devBdMesh);  

		// ... RTX Query next particle tet location
		tetQueryAccelerator.query_sync(d_particles, (double4*) d_particle_disps, d_particles_tetIDs, numParticles);

		// ... Correct floating point error when particle is close to the boundary
		if(floatErrorCorrection)
		cudaLocationCorrection(d_particles,
			d_particle_disps,
			d_particles_tetIDs,
			d_particles_triIDs,
			numParticles,
			devMesh);

#else  
		// ... Convex Query particle tet location
		convexTetQuery(devMesh, 
			d_particles,
			d_particle_disps, 
			d_particles_ConvextetIDs, 
			numParticles);

		// ... compute wall reflection
		if (reflectWall)
		convexWallReflect(devMesh, 
			d_particles_ConvextetIDs,
			d_particles, 
			d_particle_vels,
			d_particle_disps, 
			numParticles);

		//debug
		tetQueryAccelerator.query_sync(d_particles, (double4*)d_particle_disps, d_particles_tetIDs, numParticles);
		cudaReportParticles(numParticles, d_particles_tetIDs);
#endif



#ifndef  ConvexPoly
		// ... Move particles
		cudaMoveParticles(d_particles, d_particle_disps,
			numParticles, d_particles_tetIDs);
#else
		// ... Move particles
		cudaMoveParticles(d_particles, d_particle_disps,
			numParticles, d_particles_ConvextetIDs);
#endif

#ifndef  ConvexPoly
		// ... RTX Query next particle tet location
		//tetQueryAccelerator.query_sync(d_particles, (double4*) d_particle_disps, d_particles_tetIDs, numParticles);
		//tetQueryAccelerator.query_sync(d_particles, d_particles_tetIDs, numParticles);
#endif


#ifndef  ConvexPoly
//		cudaReportParticles(numParticles, d_particles_tetIDs);
#else
//		cudaReportParticles(numParticles, d_particles_ConvextetIDs);
#endif


		if (saveStreamlinetoFile)
			if ((i % (saveInterval * 1)) == 0 || i == numSteps)
				addToTrajectories(d_particles, numParticles, trajectories);

		if ((i % saveInterval) == 0 || i == numSteps)
			writeParticles2VTU(i + 1, d_particles, d_particle_vels, d_particles_tetIDs, numParticles,
				d_particles_ConvextetIDs);

		printf("------------End Step %d-------------\n\n", i);

    }
    std::cout << "#adv: advection steps = " << numSteps << std::endl;
    std::cout << "done ... ignoring proper cleanup for now" << std::endl;

#ifndef  ConvexPoly
	cudaReportParticles(numParticles, d_particles_tetIDs);
#else
	cudaReportParticles(numParticles, d_particles_ConvextetIDs);
#endif



	double runtime = timer.stop();
    printf("#adv: Simulation RunTime=%f ms\n", runtime);
	printf("#adv: Simulation Performance=%f steps/secs\n", numSteps/runtime*1000);

	if (saveStreamlinetoFile) {
		if (objTrajectoryFileName.size() > 0)
			saveTrajectories(objTrajectoryFileName, trajectories);
		if (vtkStreamlineFileName.size() > 0)
			writeStreamline2VTK(vtkStreamlineFileName, trajectories);
	}
	
	

    return 0;
  }

}
