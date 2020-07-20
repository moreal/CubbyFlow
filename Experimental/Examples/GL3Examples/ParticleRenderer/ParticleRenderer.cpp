/*************************************************************************
> File Name: ParticleRenderer.cpp
> Project Name: CubbyFlow
> This code is based on Jet Framework that was created by Doyub Kim.
> References: https://github.com/doyubkim/fluid-engine-dev
> Purpose: Particle Renderer
> Created Time: 2020/06/02
> Copyright (c) 2020, Ji-Hong snowapril
*************************************************************************/
#ifndef CUBBYFLOW_PARTICLE_RENDERER_H
#define CUBBYFLOW_PARTICLE_RENDERER_H

#include <Core/Array/Array1.h>
#include <Core/Vector/Vector2.h>
#include <Core/Vector/Vector4.h>

#include <Framework/Scene/SceneParser.h>
#include <Framework/Buffer/Framebuffer.h>
#include <Framework/Texture/Texture.h>
#include <Framework/Texture/Texture2D.h>
#include <Framework/Application/Application.h>
#include <Framework/Utils/FileSystem.h>
#include <Framework/Renderable/ScreenRenderable.h>
#include <Framework/Renderable/PointsRenderable.h>
#include <Framework/Renderable/Material.h>
#include <Framework/Shader/Shader.h>
#include <Framework/View/Viewport.h>
#include <Framework/View/Pivot.h>
#include <Framework/View/PerspectiveCamera.h>
#include <Framework/View/CameraController.h>
#include <Framework/View/Light.h>
#include <Framework/Utils/ImageLoader.h>
#include <Framework/Utils/GeometryCacheParser.h>

#include "../Utils/ClaraUtils.h"
#include "../Utils/main.h"

#include <memory>
#include <fstream>
#include <iostream>
#include <string>
#include <random>

#ifdef CUBBYFLOW_USE_GL

#include <GL3/Window/GL3Window.h>
#include <GL3/Buffer/GL3Framebuffer.h>
#include <GL3/Renderer/GL3Renderer.h>
#include <GL3/Application/GL3Application.h>
#include <GLFW/glfw3.h>

#include <Media/ScreenRecorder.h>

#ifdef CUBBYFLOW_WINDOWS
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#define APP_NAME "ParticleRenderer"

using namespace CubbyFlow;
using namespace CubbyRender;

class ParticleRenderer final : public GL3Window
{
    using super_t = GL3Window;
public: 
    //! Default constructor.
    ParticleRenderer() = default;
    //! Default Constructor.
    ParticleRenderer(const std::string& title, int width, int height, double fps);
    //! Default Destructor.
    ~ParticleRenderer();
    //! Set timer for automatic shutdown.
    void setShutdownTimer(int numFrames);
    //! Initialize the sample renderer with simulated particle files directory and format.
    bool initialize(const std::string& scenePath);
    //! Action implementation when window is resized.
    void onWindowResized(int width, int height) override;
    //! Action implementation when any key is pressed or released
    void onKey(int key, int scancode, int action, int mods) override;
    //! Action implementation when any mouse button is pressed or released.
    void onMouseButton(int button, int action, int mods) override;
    //! Action implementation when cursor is moved
    void onMouseCursorPos(double x, double y) override;
    //! Action implementation when mouse scroll is moved
    void onMouseScroll(double deltaX, double deltaY) override;
protected:
    void onRenderScene() override;
    void onUpdateScene() override;
private:
    SceneParserPtr _sceneParser;
    PointsRenderablePtr _renderable;
    GeometryCacheParserPtr _cacheParser;
    CameraControllerPtr _camController;
    double _fps = 60.0;
    int _shutdownTimer = -1;
    bool _enableCursor = false;
    bool _pause = false;
};
using ParticleRendererPtr = std::shared_ptr<ParticleRenderer>;

ApplicationPtr application;
ParticleRendererPtr renderer;

ParticleRenderer::ParticleRenderer(const std::string& title, int width, int height, double fps)
    : GL3Window(title, width, height), _fps(fps)
{
    //! Do nothing.
}

ParticleRenderer::~ParticleRenderer()
{
    _camController.reset();
    _cacheParser.reset();
    _renderable.reset();
    _sceneParser.reset();
}

bool ParticleRenderer::initialize(const std::string& scenePath)
{
    _sceneParser = std::make_shared<SceneParser>();
    _sceneParser->loadScene(scenePath); 

    CameraPtr camera = _sceneParser->getSceneObject<Camera>("camera");
    _camController = std::make_shared<CameraController>(camera);

    _cacheParser = _sceneParser->getSceneObject<GeometryCacheParser>("particleCache");
    _renderable = _sceneParser->getSceneObject<PointsRenderable>("particle");

    _renderer->addRenderable(_renderable);
    _renderer->setCamera(_camController->getCamera());
    _renderer->setBackgroundColor(Vector4F(0.1f, 0.2f, 0.4f, 1.0f));

    ImageLoader loader;
    if (!loader.loadImage("./Resources/textures/metal.png"))
        return false;


    TextureParams params; //! default setting.
    params.type = DataType::UNSIGNED_BYTE;
    params.samplingMode = ImageSamplingMode::NEAREST;

    Texture2DPtr texture = _renderer->createTexture2D(params, loader.getImageSize(), loader.data());

    MaterialPtr material = _renderer->createMaterial();
    material->addTexture(0, texture);

    ShaderPtr shader = _renderer->createShaderPreset("point_shader");
    auto& shaderParams = shader->getParameters();
    shaderParams.setParameter("sphereTexture", 0);
    material->setShader(shader);
    
    _renderable->setMaterial(material);

    return true;
}

void ParticleRenderer::setShutdownTimer(int numFrames)
{
    _shutdownTimer = numFrames;
}

void ParticleRenderer::onWindowResized(int width, int height)
{
    UNUSED_VARIABLE(width);
    UNUSED_VARIABLE(height);
}

void ParticleRenderer::onKey(int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {   
        switch(key)
        {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(_glfwWindow, GLFW_TRUE);
            break;
        case GLFW_KEY_W:
            _camController->translateCamera(-1, 0);
            break;
        case GLFW_KEY_A:
            _camController->translateCamera(0, -1);
            break;
        case GLFW_KEY_S:
            _camController->translateCamera(1, 0);
            break;
        case GLFW_KEY_D:
            _camController->translateCamera(0, 1);
            break;
        case GLFW_KEY_SPACE:
            _pause = !_pause;
            break;
        }
    }
    
    UNUSED_VARIABLE(scancode);
    UNUSED_VARIABLE(mods);
}

void ParticleRenderer::onMouseButton(int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        if (action == GLFW_PRESS)
        {
            _enableCursor = true;
        }
        else if (action == GLFW_RELEASE)
        {
            _enableCursor = false;
        }
    }

    UNUSED_VARIABLE(mods);
}   

void ParticleRenderer::onMouseCursorPos(double x, double y)
{
    static Vector2D prevMousePos = Vector2D(x, y);
    constexpr double kSensitivity = 0.1f;
    
    const double xoffset = (x - prevMousePos.x) * kSensitivity;
    const double yoffset = (prevMousePos.y - y) * kSensitivity;

    prevMousePos = Vector2D(x, y);

    if (_enableCursor)
    {
        _camController->modifyDirection(xoffset, yoffset);
    }
}   

void ParticleRenderer::onMouseScroll(double deltaX, double deltaY)
{
    UNUSED_VARIABLE(deltaX);
    UNUSED_VARIABLE(deltaY);
}

void ParticleRenderer::onRenderScene()
{
    _renderer->render();

    if (_shutdownTimer == 0)     glfwSetWindowShouldClose(_glfwWindow, GLFW_TRUE);
    else if (_shutdownTimer > 0) --_shutdownTimer;
}

void ParticleRenderer::onUpdateScene()
{
    if (_pause) return;
    static size_t count = 0;

    _renderable->update(_cacheParser->getVertexCache(count));
    if (++count >= _cacheParser->getNumberOfGeometryCache())
        count = 0;

    _camController->orbitRotation(Vector3F(0.0f, 0.0f, 0.0f), 0.02f, 0.0f, 3.0f);
}

int sampleMain(int argc, const char** argv)
{
    bool showHelp = false;
    int numberOfFrames = 60;
    int resX = 800;
    int resY = 600;
    double fps = 60.0;
    std::string logFileName = APP_NAME ".log";
    std::string scenePath;
    std::string outputDir = APP_NAME "_output";

    // Parsing
    auto parser =
        clara::Help(showHelp) |
        clara::Opt(numberOfFrames, "numberOfFrames")
        ["-n"]["--numframe"]
        ("number of frames will be rendered (default is -1 which means [no shutdown timer])") |
        clara::Opt(resX, "resX")
        ["-x"]["--resx"]
        ("grid resolution in x-axis (default is 800)") |
        clara::Opt(resY, "resY")
        ["-y"]["--resy"]
        ("grid resolution in y-axis (default is 600)") |
        clara::Opt(fps, "fps")
        ["-f"]["--fps"]
        ("frame per second for simulation (default is 60)") |
        clara::Opt(logFileName, "logFileName")
        ["-l"]["--log"]
        ("log file name (default is " APP_NAME ".log)") |
        clara::Opt(scenePath, "scenePath")
        ["-s"]["--scenePath"]
        ("path to scene file for parsing scene objects.") |
        clara::Opt(outputDir, "outputDir")
        ["-o"]["--output"]
        ("output directory name (default is " APP_NAME "_output)");

    auto result = parser.parse(clara::Args(argc, argv));
    if (!result)
    {
        std::cerr << "Error in command line: " << result.errorMessage() << '\n';
        exit(EXIT_FAILURE);
    }

    if (scenePath.empty() || showHelp)
    {
        std::cout << ToString(parser) << '\n';
        exit(EXIT_SUCCESS);
    }

#ifdef CUBBYFLOW_WINDOWS
    _mkdir(outputDir.c_str());
#else
    mkdir(outputDir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif

    std::ofstream logFile(logFileName.c_str());
    if (logFile)
    {
        Logging::SetAllStream(&logFile);
    }

    application = std::make_shared<GL3Application>();
    if (application->initialize())
    {
        std::cerr << "Initialize application failed." << std::endl;
        return -1;
    }
    
    renderer = std::make_shared<ParticleRenderer>(APP_NAME, resX, resY, fps);
    if (renderer->initialize(scenePath) == false)
    {
        std::cerr << "Renderer Window Initialize Failed" << std::endl;
        return -1;
    }
    renderer->setShutdownTimer(numberOfFrames);
    
    int exitCode = 0;
    application->setMainWindow(renderer);

#ifdef CUBBYFLOW_RECORDING
    auto recorder = std::make_shared<ScreenRecorder>();
    recorder->setWorkingDirectory(APP_NAME "_output");
    exitCode = application->run([&](Size2 dim, const ArrayAccessor1<unsigned char>& frame){
        return recorder->EncodeFrame(dim, frame);
    });
#else
    exitCode = application->run();
#endif

    //! release application and renderer window .
    renderer.reset();
    application.reset();
    return exitCode;
}

#endif
#endif 