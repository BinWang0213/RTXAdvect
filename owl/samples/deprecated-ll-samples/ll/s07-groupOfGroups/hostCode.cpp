// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
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

// public owl-ll API
#include <owl/ll.h>
// our device-side data structures
#include "GeomTypes.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <random>

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#ll.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#ll.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

const char *outFileName = "ll07-groupOfGroups.png";
const vec2i fbSize(1600,800);
const vec3f lookFrom(13, 2, 3);
const vec3f lookAt(0, 0, 0);
const vec3f lookUp(0.f,1.f,0.f);
const float fovy = 20.f;

std::vector<DielectricSphere> dielectricSpheres;
std::vector<LambertianSphere> lambertianSpheres;
std::vector<MetalSphere>      metalSpheres;

struct {
  std::vector<vec3f> vertices;
  std::vector<vec3i> indices;
  std::vector<Dielectric> materials;
} dielectricBoxes;
struct {
  std::vector<vec3f> vertices;
  std::vector<vec3i> indices;
  std::vector<Metal> materials;
} metalBoxes;
struct {
  std::vector<vec3f> vertices;
  std::vector<vec3i> indices;
  std::vector<Lambertian> materials;
} lambertianBoxes;

inline size_t max3(size_t a, size_t b, size_t c)
{ return std::max(std::max(a,b),c); }

inline float rnd()
{
  static std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

inline vec3f rnd3f() { return vec3f(rnd(),rnd(),rnd()); }

inline vec3f randomPointInUnitSphere() {
  vec3f p;
  do {
    p = 2.f*vec3f(rnd(),rnd(),rnd()) - vec3f(1.f);
  } while (dot(p,p) >= 1.f);
  return p;
}


template<typename BoxArray, typename Material>
void addRandomBox(BoxArray &boxes,
                  const vec3f &center,
                  const float size,
                  const Material &material)
{
  const int NUM_VERTICES = 8;
  static const vec3f unitBoxVertices[NUM_VERTICES] =
    {
      {-1.f, -1.f, -1.f},
      {+1.f, -1.f, -1.f},
      {+1.f, +1.f, -1.f},
      {-1.f, +1.f, -1.f},
      {-1.f, +1.f, +1.f},
      {+1.f, +1.f, +1.f},
      {+1.f, -1.f, +1.f},
      {-1.f, -1.f, +1.f},
    };

  const int NUM_INDICES = 12;
  static const vec3i unitBoxIndices[NUM_INDICES] =
    {
      {0, 2, 1}, //face front
      {0, 3, 2},
      {2, 3, 4}, //face top
      {2, 4, 5},
      {1, 2, 5}, //face right
      {1, 5, 6},
      {0, 7, 4}, //face left
      {0, 4, 3},
      {5, 4, 7}, //face back
      {5, 7, 6},
      {0, 6, 7}, //face bottom
      {0, 1, 6}
    };

  const vec3f U = normalize(randomPointInUnitSphere());
  owl::affine3f xfm = owl::frame(U);
  xfm = owl::affine3f(owl::linear3f::rotate(U,rnd())) * xfm;
  xfm = owl::affine3f(owl::linear3f::scale(.7f*size)) * xfm;
  xfm = owl::affine3f(owl::affine3f::translate(center)) * xfm;
  
  const int startIndex = (int)boxes.vertices.size();
  for (int i=0;i<NUM_VERTICES;i++)
    boxes.vertices.push_back(owl::xfmPoint(xfm,unitBoxVertices[i]));
  for (int i=0;i<NUM_INDICES;i++)
    boxes.indices.push_back(unitBoxIndices[i]+vec3i(startIndex));
  boxes.materials.push_back(material);
}

void createScene()
{
  lambertianSpheres.push_back({Sphere{vec3f(0.f, -1000.0f, -1.f), 1000.f},
        Lambertian{vec3f(0.5f, 0.5f, 0.5f)}});
  
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      float choose_shape = rnd();
      vec3f center(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) {
        if (choose_shape > .5f) {
          addRandomBox(lambertianBoxes,center,.2f,
                       Lambertian{rnd3f()*rnd3f()});
        } else
          lambertianSpheres.push_back({Sphere{center, 0.2f},
                Lambertian{rnd3f()*rnd3f()}});
      } else if (choose_mat < 0.95f) {
        if (choose_shape > .5f) {
          addRandomBox(metalBoxes,center,.2f,
                       Metal{0.5f*(1.f+rnd3f()),0.5f*rnd()});
        } else
          metalSpheres.push_back({Sphere{center, 0.2f},
                Metal{0.5f*(1.f+rnd3f()),0.5f*rnd()}});
      } else {
        if (choose_shape > .5f) {
          addRandomBox(dielectricBoxes,center,.2f,
                       Dielectric{1.5f});
        } else
          dielectricSpheres.push_back({Sphere{center, 0.2f},
                Dielectric{1.5f}});
      }
    }
  }
  dielectricSpheres.push_back({Sphere{vec3f(0.f, 1.f, 0.f), 1.f},
        Dielectric{1.5f}});
  lambertianSpheres.push_back({Sphere{vec3f(-4.f,1.f, 0.f), 1.f},
        Lambertian{vec3f(0.4f, 0.2f, 0.1f)}});
  metalSpheres.push_back({Sphere{vec3f(4.f, 1.f, 0.f), 1.f},
        Metal{vec3f(0.7f, 0.6f, 0.5f), 0.0f}});
}
  
int main(int ac, char **av)
{
  LOG("ll example '" << av[0] << "' starting up");

  LOG("creating the scene ...");
  createScene();
  LOG_OK("created scene:");
  LOG_OK(" num lambertian spheres: " << lambertianSpheres.size());
  LOG_OK(" num dielectric spheres: " << dielectricSpheres.size());
  LOG_OK(" num metal spheres     : " << metalSpheres.size());
  
  LLOContext llo = lloContextCreate(nullptr,0);

  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################
  LOG("building pipeline ...");
  lloAllocModules(llo,1);
  lloModuleCreate(llo,0,ptxCode);
  lloBuildModules(llo);
  
  enum { METAL_SPHERES_TYPE=0,
         DIELECTRIC_SPHERES_TYPE,
         LAMBERTIAN_SPHERES_TYPE,
         METAL_BOXES_TYPE,
         DIELECTRIC_BOXES_TYPE,
         LAMBERTIAN_BOXES_TYPE,
         NUM_GEOM_TYPES };
  lloAllocGeomTypes(llo,NUM_GEOM_TYPES);
  lloGeomTypeCreate(llo,METAL_SPHERES_TYPE,sizeof(MetalSpheresGeom));
  lloGeomTypeCreate(llo,LAMBERTIAN_SPHERES_TYPE,sizeof(LambertianSpheresGeom));
  lloGeomTypeCreate(llo,DIELECTRIC_SPHERES_TYPE,sizeof(DielectricSpheresGeom));
  
  lloGeomTypeCreate(llo,METAL_BOXES_TYPE,sizeof(MetalBoxesGeom));
  lloGeomTypeCreate(llo,LAMBERTIAN_BOXES_TYPE,sizeof(LambertianBoxesGeom));
  lloGeomTypeCreate(llo,DIELECTRIC_BOXES_TYPE,sizeof(DielectricBoxesGeom));
  
  lloGeomTypeClosestHit(llo,/*geom type ID*/LAMBERTIAN_SPHERES_TYPE,
                        /*ray type  */0,
                        /*module:*/0,
                        "LambertianSpheres");
  lloGeomTypeIntersect(llo,/*geom type ID*/LAMBERTIAN_SPHERES_TYPE,
                       /*ray type  */0,
                       /*module:*/0,
                       "LambertianSpheres");
  lloGeomTypeBoundsProgDevice(llo,/*program ID*/LAMBERTIAN_SPHERES_TYPE,
                                  /*module:*/0,
                                  "LambertianSpheres",
                                  sizeof(LambertianSpheresGeom));
  
  lloGeomTypeClosestHit(llo,/*geom type ID*/DIELECTRIC_SPHERES_TYPE,
                        /*ray type  */0,
                        /*module:*/0,
                        "DielectricSpheres");
  lloGeomTypeIntersect(llo,/*geom type ID*/DIELECTRIC_SPHERES_TYPE,
                       /*ray type  */0,
                       /*module:*/0,
                       "DielectricSpheres");
  lloGeomTypeBoundsProgDevice(llo,/*program ID*/DIELECTRIC_SPHERES_TYPE,
                                  /*module:*/0,
                                  "DielectricSpheres",
                                  sizeof(DielectricSpheresGeom));
  
  lloGeomTypeClosestHit(llo,/*geom type ID*/METAL_SPHERES_TYPE,
                        /*ray type  */0,
                        /*module:*/0,
                        "MetalSpheres");
  lloGeomTypeIntersect(llo,/*geom type ID*/METAL_SPHERES_TYPE,
                       /*ray type  */0,
                       /*module:*/0,
                       "MetalSpheres");
  lloGeomTypeBoundsProgDevice(llo,/*program ID*/METAL_SPHERES_TYPE,
                                  /*module:*/0,
                                  "MetalSpheres",
                                  sizeof(MetalSpheresGeom));

  // now for the box types - thos use triangles, so already have
  // bounds and isec, only need closesthit
  lloGeomTypeClosestHit(llo,/*geom type ID*/LAMBERTIAN_BOXES_TYPE,
                        /*ray type  */0,
                        /*module:*/0,
                        "LambertianBoxes");
  lloGeomTypeClosestHit(llo,/*geom type ID*/DIELECTRIC_BOXES_TYPE,
                        /*ray type  */0,
                        /*module:*/0,
                        "DielectricBoxes");
  lloGeomTypeClosestHit(llo,/*geom type ID*/METAL_BOXES_TYPE,
                        /*ray type  */0,
                        /*module:*/0,
                        "MetalBoxes");



  lloAllocRayGens(llo,1);
  lloRayGenCreate(llo,/*program ID*/0,
                  /*module:*/0,
                  "rayGen",
                  sizeof(RayGenData));
  
  lloAllocMissProgs(llo,1);
  lloMissProgCreate(llo,/*program ID*/0,
                    /*module:*/0,
                    "miss",
                    sizeof(MissProgData));
  lloBuildPrograms(llo);
  lloCreatePipeline(llo);

  LOG("building geometries ...");

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  enum { FRAME_BUFFER=0,
         LAMBERTIAN_SPHERES_BUFFER,
         DIELECTRIC_SPHERES_BUFFER,
         METAL_SPHERES_BUFFER,

         LAMBERTIAN_BOXES_MATERIAL_BUFFER,
         DIELECTRIC_BOXES_MATERIAL_BUFFER,
         METAL_BOXES_MATERIAL_BUFFER,

         LAMBERTIAN_BOXES_VERTEX_BUFFER,
         DIELECTRIC_BOXES_VERTEX_BUFFER,
         METAL_BOXES_VERTEX_BUFFER,

         LAMBERTIAN_BOXES_INDEX_BUFFER,
         DIELECTRIC_BOXES_INDEX_BUFFER,
         METAL_BOXES_INDEX_BUFFER,

         NUM_BUFFERS };
  lloAllocBuffers(llo,NUM_BUFFERS);
  lloHostPinnedBufferCreate(llo,FRAME_BUFFER,fbSize.x*fbSize.y*sizeof(uint32_t));

  // ------------------------------------------------------------------
  // alloc geom
  // ------------------------------------------------------------------
  enum { LAMBERTIAN_SPHERES_GEOM=0,
         DIELECTRIC_SPHERES_GEOM,
         METAL_SPHERES_GEOM,
         LAMBERTIAN_BOXES_GEOM,
         DIELECTRIC_BOXES_GEOM,
         METAL_BOXES_GEOM,
         NUM_GEOMS };
  lloAllocGeoms(llo,NUM_GEOMS);

  // ----------- the spheres -----------
  lloUserGeomCreate(llo,/* geom ID    */LAMBERTIAN_SPHERES_GEOM,
                     /* type/PG ID */LAMBERTIAN_SPHERES_TYPE,
                     /* numprims   */lambertianSpheres.size());
  lloDeviceBufferCreate(llo,LAMBERTIAN_SPHERES_BUFFER,
                        lambertianSpheres.size()
                        *sizeof(lambertianSpheres[0]),
                        lambertianSpheres.data());
  lloUserGeomCreate(llo,/* geom ID    */DIELECTRIC_SPHERES_GEOM,
                     /* type/PG ID */DIELECTRIC_SPHERES_TYPE,
                     /* numprims   */dielectricSpheres.size());
  lloDeviceBufferCreate(llo,DIELECTRIC_SPHERES_BUFFER,
                        dielectricSpheres.size()
                        *sizeof(dielectricSpheres[0]),
                        dielectricSpheres.data());
  lloUserGeomCreate(llo,/* geom ID    */METAL_SPHERES_GEOM,
                     /* type/PG ID */METAL_SPHERES_TYPE,
                     /* numprims   */metalSpheres.size());
  lloDeviceBufferCreate(llo,METAL_SPHERES_BUFFER,
                        metalSpheres.size()
                        *sizeof(metalSpheres[0]),
                        metalSpheres.data());

  // ----------- the boxes -----------
  lloTrianglesGeomCreate(llo,/* geom ID    */LAMBERTIAN_BOXES_GEOM,
                          /* type/PG ID */LAMBERTIAN_BOXES_TYPE);
  lloTrianglesGeomCreate(llo,/* geom ID    */DIELECTRIC_BOXES_GEOM,
                          /* type/PG ID */DIELECTRIC_BOXES_TYPE);
  lloTrianglesGeomCreate(llo,/* geom ID    */METAL_BOXES_GEOM,
                          /* type/PG ID */METAL_BOXES_TYPE);

  // indices
  LOG("creating index buffers");
  lloDeviceBufferCreate(llo,LAMBERTIAN_BOXES_INDEX_BUFFER,
                        lambertianBoxes.indices.size()
                        *sizeof(lambertianBoxes.indices[0]),
                        lambertianBoxes.indices.data());
  lloDeviceBufferCreate(llo,DIELECTRIC_BOXES_INDEX_BUFFER,
                        dielectricBoxes.indices.size()
                        *sizeof(dielectricBoxes.indices[0]),
                        dielectricBoxes.indices.data());
  lloDeviceBufferCreate(llo,METAL_BOXES_INDEX_BUFFER,
                        metalBoxes.indices.size()
                        *sizeof(metalBoxes.indices[0]),
                        metalBoxes.indices.data());
  // vertices
  LOG("creating vertex buffers");
  lloDeviceBufferCreate(llo,LAMBERTIAN_BOXES_VERTEX_BUFFER,
                        lambertianBoxes.vertices.size()
                        *sizeof(lambertianBoxes.vertices[0]),
                        lambertianBoxes.vertices.data());
  lloDeviceBufferCreate(llo,DIELECTRIC_BOXES_VERTEX_BUFFER,
                        dielectricBoxes.vertices.size()
                        *sizeof(dielectricBoxes.vertices[0]),
                        dielectricBoxes.vertices.data());
  lloDeviceBufferCreate(llo,METAL_BOXES_VERTEX_BUFFER,
                        metalBoxes.vertices.size()
                        *sizeof(metalBoxes.vertices[0]),
                        metalBoxes.vertices.data());
  // materials
  LOG("creating box material buffers");
  lloDeviceBufferCreate(llo,LAMBERTIAN_BOXES_MATERIAL_BUFFER,
                        lambertianBoxes.materials.size()
                        *sizeof(lambertianBoxes.materials[0]),
                        lambertianBoxes.materials.data());
  lloDeviceBufferCreate(llo,DIELECTRIC_BOXES_MATERIAL_BUFFER,
                        dielectricBoxes.materials.size()
                        *sizeof(dielectricBoxes.materials[0]),
                        dielectricBoxes.materials.data());
  lloDeviceBufferCreate(llo,METAL_BOXES_MATERIAL_BUFFER,
                        metalBoxes.materials.size()
                        *sizeof(metalBoxes.materials[0]),
                        metalBoxes.materials.data());

  // ##################################################################
  // set triangle mesh vertex/index buffers
  // ##################################################################
  lloTrianglesGeomSetVertexBuffer
    (llo,
     /* geom ID   */LAMBERTIAN_BOXES_GEOM,
     /* buffer ID */LAMBERTIAN_BOXES_VERTEX_BUFFER,
     /* meta info */lambertianBoxes.vertices.size(),sizeof(vec3f),0);
  lloTrianglesGeomSetIndexBuffer
    (llo,
     /* geom ID   */LAMBERTIAN_BOXES_GEOM,
     /* buffer ID */LAMBERTIAN_BOXES_INDEX_BUFFER,
     /* meta info */lambertianBoxes.indices.size(),sizeof(vec3i),0);

  lloTrianglesGeomSetVertexBuffer
    (llo,
     /* geom ID   */METAL_BOXES_GEOM,
     /* buffer ID */METAL_BOXES_VERTEX_BUFFER,
     /* meta info */metalBoxes.vertices.size(),sizeof(vec3f),0);
  lloTrianglesGeomSetIndexBuffer
    (llo,
     /* geom ID   */METAL_BOXES_GEOM,
     /* buffer ID */METAL_BOXES_INDEX_BUFFER,
     /* meta info */metalBoxes.indices.size(),sizeof(vec3i),0);

  lloTrianglesGeomSetVertexBuffer
    (llo,
     /* geom ID   */DIELECTRIC_BOXES_GEOM,
     /* buffer ID */DIELECTRIC_BOXES_VERTEX_BUFFER,
     /* meta info */dielectricBoxes.vertices.size(),sizeof(vec3f),0);
  lloTrianglesGeomSetIndexBuffer
    (llo,
     /* geom ID   */DIELECTRIC_BOXES_GEOM,
     /* buffer ID */DIELECTRIC_BOXES_INDEX_BUFFER,
     /* meta info */dielectricBoxes.indices.size(),sizeof(vec3i),0);
  
  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################

  enum { SPHERES_GROUP=0,
         BOXES_GROUP,
         WORLD_GROUP,
         NUM_GROUPS };
  lloAllocGroups(llo,NUM_GROUPS);

  // ----------- first, the spheres group -----------
  int geomsInSpheresGroup[] =
    { LAMBERTIAN_SPHERES_GEOM,
      DIELECTRIC_SPHERES_GEOM,
      METAL_SPHERES_GEOM };
  lloUserGeomGroupCreate(llo,/* group ID */SPHERES_GROUP,
                         /* geoms in group, pointer */ geomsInSpheresGroup,
                         /* geoms in group, count   */ 3);
  lloGroupBuildPrimitiveBounds
    (llo,SPHERES_GROUP,max3(sizeof(MetalSpheresGeom),
                        sizeof(DielectricSpheresGeom),
                        sizeof(LambertianSpheresGeom)),
     [&](uint8_t *output, int devID, int geomID, int childID) {
      switch(geomID) {
      case LAMBERTIAN_SPHERES_GEOM:
        ((LambertianSpheresGeom*)output)->prims
          = (LambertianSphere*)lloBufferGetPointer(llo,LAMBERTIAN_SPHERES_BUFFER,devID);
        break;
      case DIELECTRIC_SPHERES_GEOM:
        ((DielectricSpheresGeom*)output)->prims
          = (DielectricSphere*)lloBufferGetPointer(llo,DIELECTRIC_SPHERES_BUFFER,devID);
        break;
      case METAL_SPHERES_GEOM:
        ((MetalSpheresGeom*)output)->prims
          = (MetalSphere*)lloBufferGetPointer(llo,METAL_SPHERES_BUFFER,devID);
        break;
      default:
        assert(0);
      }
    });
  lloGroupAccelBuild(llo,SPHERES_GROUP);

  // ----------- now, the boxes group -----------
  int geomsInBoxesGroup[]
    = { LAMBERTIAN_BOXES_GEOM,
        DIELECTRIC_BOXES_GEOM,
        METAL_BOXES_GEOM };
  lloTrianglesGeomGroupCreate(llo,/* group ID */BOXES_GROUP,
                              /* geoms in group, pointer */ geomsInBoxesGroup,
                              /* geoms in group, count   */ 3);
  lloGroupAccelBuild(llo,BOXES_GROUP);

  
  // ----------- finally, the world group that combines them -----------
  const owl::affine3f unitXfm = owl::one;
#if TEST_CASE==1
  int groupsInWorldGroup[]
    = { SPHERES_GROUP,
        BOXES_GROUP };
  lloInstanceGroupCreate(llo,
                         /* group ID */WORLD_GROUP,
                         /* geoms in group, pointer */ groupsInWorldGroup,
                         /* geoms in group, count   */ 2);
  lloInstanceGroupSetTransform(llo,WORLD_GROUP,0,
                               (const float*)&unitXfm);
  lloInstanceGroupSetTransform(llo,WORLD_GROUP,1,
                               (const float*)&unitXfm);
#elif TEST_CASE==2
  lloInstanceGroupCreate(llo,
                         /* group ID */WORLD_GROUP,
                         /* geoms in group, pointer */ nullptr,
                         /* geoms in group, count   */ 2);
  lloInstanceGroupSetChild(llo,WORLD_GROUP,0,
                           SPHERES_GROUP);
  lloInstanceGroupSetChild(llo,WORLD_GROUP,1,
                           BOXES_GROUP);
#else
  int groupsInWorldGroup[]
    = { SPHERES_GROUP,
        BOXES_GROUP };
  lloInstanceGroupCreate(llo,
                         /* group ID */WORLD_GROUP,
                         /* geoms in group, pointer */ groupsInWorldGroup,
                         /* geoms in group, count   */ 2);
#endif
  lloGroupAccelBuild(llo,WORLD_GROUP);
  
  
  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################
  LOG("building SBT ...");

  // ----------- build hitgroups -----------
  lloSbtHitProgsBuild
    (llo,
     [&](uint8_t *output,int devID,int geomID,int childID) {
      switch(geomID) {
      case LAMBERTIAN_SPHERES_GEOM:
        ((LambertianSpheresGeom*)output)->prims
          = (LambertianSphere*)lloBufferGetPointer(llo,LAMBERTIAN_SPHERES_BUFFER,devID);
        break;
      case DIELECTRIC_SPHERES_GEOM:
        ((DielectricSpheresGeom*)output)->prims
          = (DielectricSphere*)lloBufferGetPointer(llo,DIELECTRIC_SPHERES_BUFFER,devID);
        break;
      case METAL_SPHERES_GEOM:
        ((MetalSpheresGeom*)output)->prims
          = (MetalSphere*)lloBufferGetPointer(llo,METAL_SPHERES_BUFFER,devID);
        break;

      case LAMBERTIAN_BOXES_GEOM: {
        LambertianBoxesGeom &self = *(LambertianBoxesGeom*)output;
        self.index
          = (vec3i*)lloBufferGetPointer(llo,LAMBERTIAN_BOXES_INDEX_BUFFER,devID);
        self.vertex
          = (vec3f*)lloBufferGetPointer(llo,LAMBERTIAN_BOXES_VERTEX_BUFFER,devID);
        self.perBoxMaterial
          = (Lambertian *)lloBufferGetPointer(llo,LAMBERTIAN_BOXES_MATERIAL_BUFFER,devID);
      } break;
      case DIELECTRIC_BOXES_GEOM: {
        DielectricBoxesGeom &self = *(DielectricBoxesGeom*)output;
        self.index
          = (vec3i*)lloBufferGetPointer(llo,DIELECTRIC_BOXES_INDEX_BUFFER,devID);
        self.vertex
          = (vec3f*)lloBufferGetPointer(llo,DIELECTRIC_BOXES_VERTEX_BUFFER,devID);
        self.perBoxMaterial
          = (Dielectric *)lloBufferGetPointer(llo,DIELECTRIC_BOXES_MATERIAL_BUFFER,devID);
      } break;
      case METAL_BOXES_GEOM: {
        MetalBoxesGeom &self = *(MetalBoxesGeom*)output;
        self.index
          = (vec3i*)lloBufferGetPointer(llo,METAL_BOXES_INDEX_BUFFER,devID);
        self.vertex
          = (vec3f*)lloBufferGetPointer(llo,METAL_BOXES_VERTEX_BUFFER,devID);
        self.perBoxMaterial
          = (Metal *)lloBufferGetPointer(llo,METAL_BOXES_MATERIAL_BUFFER,devID);
      } break;
      default:
        assert(0);
      }
    });
  
  // ----------- build miss prog(s) -----------
  lloSbtMissProgsBuild
    (llo,
     [&](uint8_t *output,
         int devID,
         int rayType) {
      /* we don't have any ... */
    });
  
  // ----------- build raygens -----------
  lloSbtRayGensBuild
    (llo,
     [&](uint8_t *output,
         int devID,
         int rgID) {
      RayGenData *rg = (RayGenData*)output;
      rg->deviceIndex   = devID;
      rg->deviceCount = lloGetDeviceCount(llo);
      rg->fbSize = fbSize;
      rg->fbPtr  = (uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,devID);
      rg->worldAccel     = lloGroupGetTraversable(llo,WORLD_GROUP,devID);
      rg->worldSBTOffset = lloGroupGetSbtOffset(llo,WORLD_GROUP);

      const float vfov = fovy;
      const vec3f vup = lookUp;
      const float aspect = fbSize.x / float(fbSize.y);
      const float theta = vfov * ((float)M_PI) / 180.0f;
      const float half_height = tanf(theta / 2.0f);
      const float half_width = aspect * half_height;
      const float aperture = 0.f;
      const float focusDist = 10.f;
      const vec3f origin = lookFrom;
      const vec3f w = normalize(lookFrom - lookAt);
      const vec3f u = normalize(cross(vup, w));
      const vec3f v = cross(w, u);
      const vec3f lower_left_corner
        = origin - half_width * focusDist*u - half_height * focusDist*v - focusDist * w;
      const vec3f horizontal = 2.0f*half_width*focusDist*u;
      const vec3f vertical = 2.0f*half_height*focusDist*v;

      rg->camera.origin = origin;
      rg->camera.lower_left_corner = lower_left_corner;
      rg->camera.horizontal = horizontal;
      rg->camera.vertical = vertical;
    });
  LOG_OK("everything set up ...");

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  
  LOG("trying to launch ...");
  lloLaunch2D(llo,0,fbSize.x,fbSize.y);
  // todo: explicit sync?
  
  LOG("done with launch, writing picture ...");
  // for host pinned mem it doesn't matter which device we query...
  const uint32_t *fb = (const uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,0);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
  LOG("destroying devicegroup ...");
  lloContextDestroy(llo);
  
  LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
