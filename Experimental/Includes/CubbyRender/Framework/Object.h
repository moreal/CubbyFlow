/*************************************************************************
> File Name: Object.h
> Project Name: CubbyFlow
> This code is based on Jet Framework that was created by Doyub Kim.
> References: https://github.com/doyubkim/fluid-engine-dev
> Purpose: Abstract base class for Renderer Object
> Created Time: 2020/02/19
> Copyright (c) 2020, Ji-Hong snowapril
*************************************************************************/
#ifndef CUBBYFLOW_OBJECT_H
#define CUBBYFLOW_OBJECT_H

#include <memory>

namespace CubbyFlow {
namespace CubbyRender {
    
    class Renderer;
    using RendererPtr = std::shared_ptr<Renderer>;

    //!
    //! \brief Abstract base class for Renderer Object
    //!
    class Object
    {
    public:
        //! Default constructor.
        Object();

        //! Construct with shader name (for example, shadow_shader, etc..)
        Object(const std::string& shaderName);

        //! Default destructor.
        virtual ~Object();

        //! Bind this shader to renderer
        void bind(RendererPtr renderer);

        //! Unbind this shader from renderer
        void unbind(RendererPtr renderer);    

        //! Destroy this shader instance
        void destroy();
    protected:
        //! implementation of bind method
        virtual void onBind(RendererPtr renderer) = 0;
        
        //! implementation of unbind method
        virtual void onUnbind(RendererPtr renderer) = 0;

        //! implementation of destry method
        virtual void onDestroy() = 0;
    private:
    };

    using ObjectPtr = std::shared_ptr<Object>;
} 
}

#endif