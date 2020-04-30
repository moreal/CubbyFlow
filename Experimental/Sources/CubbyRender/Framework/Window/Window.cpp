/*************************************************************************
> File Name: Window.cpp
> Project Name: CubbyFlow
> This code is based on Jet Framework that was created by Doyub Kim.
> References: https://github.com/doyubkim/fluid-engine-dev
> Purpose: Abstract base class for WIndow instance.
> Created Time: 2020/02/18
> Copyright (c) 2020, Ji-Hong snowapril
*************************************************************************/

#include <Framework/Window/Window.h>
#include <Framework/Renderer/Renderer.h>
#include <Framework/Utils/Common.h>
#include <Framework/View/CameraController.h>
#include <Framework/Simulation/Simulation.h>
#include <Core/Point/Point2.h>

namespace CubbyFlow {
namespace CubbyRender {
    
    Window::Window()
    {
        //! Do nothing.
    }

    Window::Window(const std::string& title, int width, int height)
        : _title(title), _windowSize(width, height)
    {
        //! Do nothing.
    }

    Window::~Window()
    {
        //! Do nothing.
    }

    Vector2F Window::displayScalingFactor() const
    {
        Point2F factor = (getFramebufferSize().CastTo<float>() / getWindowSize().CastTo<float>());
        return Vector2F { factor.x, factor.y };
    }

    void Window::requestRender(unsigned int numFrames)
    {
        //! Do nothing.
        UNUSED_VARIABLE(numFrames);
    }

    void Window::setSwapInterval(int interval)
    {
        _swapInterval = interval;
    }

    const RendererPtr& Window::getRenderer() const
    {
        return _renderer;
    }

    bool Window::isUpdateEnabled() const
    {
        return _isUpdateEnabled;
    }

    void Window::setIsUpdateEnabled(bool enabled)
    {
        _isUpdateEnabled = enabled;
    }

    std::string Window::getTitle()
    {
        return _title;
    }

    void Window::setRenderer(RendererPtr renderer)
    {
        _renderer = renderer;
    }

    RendererPtr Window::getRenderer()
    {
        return _renderer;
    }

    void Window::render()
    {
        _renderer->setViewport(0, 0, static_cast<size_t>(_windowSize[0] / 2.0f), static_cast<size_t>(_windowSize[1] / 2.0f));
        _renderer->render();
    }

    void Window::update()
    {
        if (_simulations.empty() == false)
            _simulations[_currentSimulationIndex]->advanceSimulation();
        
        onUpdate();
    }

    void Window::switchSimulation(int index)
    {
        //! Reset simulation which will be postponed.
        _simulations[_currentSimulationIndex]->resetSimulation();
        const int numSimulation = static_cast<int>(_simulations.size());
        
        _currentSimulationIndex += index;
        if (_currentSimulationIndex >= 0)
            _currentSimulationIndex = _currentSimulationIndex % numSimulation;
        else
            _currentSimulationIndex = numSimulation - (-_currentSimulationIndex % numSimulation);
    }

    void Window::addSimulation(SimulationPtr simulation)
    {
        simulation->setup(shared_from_this());
        _simulations.push_back(simulation);
    }

    void Window::setCameraController(CameraControllerPtr camController)
    {
        _camController = camController;
        _renderer->setCamera(_camController->getCamera());
    }
} 
}