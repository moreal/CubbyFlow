/*************************************************************************
> File Name: GL3Renderer.h
> Project Name: CubbyFlow
> This code is based on Jet Framework that was created by Doyub Kim.
> References: https://github.com/doyubkim/fluid-engine-dev
> Purpose: Renderer class implemented with Modern OpenGL
> Created Time: 2020/02/18
> Copyright (c) 2020, Ji-Hong snowapril
*************************************************************************/
#ifndef CUBBYFLOW_GLRENDERER_H
#define CUBBYFLOW_GLRENDERER_H

#ifdef CUBBYFLOW_USE_GL

#include <Framework/Renderer/Renderer.h>
#include <string>
#include <memory>

namespace CubbyFlow {
namespace CubbyRender {
    
    //!
    //! \brief Renderer interface implemeneted by Modern opengl(exactly above opengl3.3)
    //!
    class GL3Renderer final : public Renderer
    {
    public:
        //! Default constructor.
        GL3Renderer();

        //! Default destructor.
        ~GL3Renderer();

        //! let inputlayout draw it's vertices using this renderer instance.
        void draw(InputLayoutPtr inputLayout) override;

        //! Initialize and fetch gl commands.
        int initializeGL() override;

        //! get current frame image as array of floating point data.
        ArrayAccessor1<unsigned char> getCurrentFrame(Size2 size) override;

        //! Create Inputlayout pointer with default constructor.
        InputLayoutPtr createInputLayout() override;

        //! Create VertexBuffer pointer with given parameters.
        //!
        //! \param inputLayout Input Layout instance which will contain generated vertex buffer.
        //! \param material material which contains shader 
        //! \param data vertices data.
        //! \param numberOfVertices number of vertex in the data.
        //! \param format format of the input vertex
        //! \return new vertex buffer instance
        VertexBufferPtr createVertexBuffer(const ConstArrayAccessor1<float>& data, size_t numberOfVertices, VertexFormat format) override;

        //! Create VertexBuffer pointer with given parameters.
        //!
        //! \param inputLayout Input Layout instance which will contain generated index buffer.
        //! \param material material which contains shader 
        //! \param data indices data.
        //! \param numberOfIndices number of vertex in the data.
        //! \return new index buffer instance
        IndexBufferPtr createIndexBuffer(const ConstArrayAccessor1<unsigned int>& data, size_t numberOfIndices) override;

        //! Create Shader Program from presets.
        //! \param shader preset name
        //! \return new shader pointer
        ShaderPtr createShaderPreset(const std::string& shaderName) override;
        
        //! Set viewport of the current window
        //! \param x left top x position
        //! \param y left top y position
        //! \param width width of the viewport
        //! \param height height of the viewport.
        void setViewport(int x, int y, size_t width, size_t height) override;
    protected:
        void onRenderBegin() override;
        void onRenderEnd() override;
        void onSetRenderState() override;
    private:
    };

    using GL3RendererPtr = std::shared_ptr<GL3Renderer>;
} 
}

#endif

#endif 