// -*- mode: c++ -*-

#pragma once

// include glew before include gl and glfw !!
# include <GL/glew.h>

#if defined(__APPLE__) || defined(MACOSX)
# include <GLFW/glfw3.h>
#else
# define GL_GLEXT_PROTOTYPES
# include <GLFW/glfw3.h>
#endif

# include <assimp/scene.h>
# include <assimp/mesh.h>
# include <vector>

class Mesh
{
public :
	struct MeshEntry {
          enum BUFFERS {
            VERTEX_BUFFER, TEXCOORD_BUFFER, NORMAL_BUFFER, INDEX_BUFFER
          };
          GLuint vao;
          GLuint vbo[4];

          unsigned int elementCount;

          MeshEntry(aiMesh *mesh);
          ~MeshEntry();

          void load(aiMesh *mesh);
          void render();
	};

  std::vector<MeshEntry*> meshEntries;

public:
  Mesh(const char *filename);
  ~Mesh(void);

  void render();
};

