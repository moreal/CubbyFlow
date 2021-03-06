// This code is based on Jet framework.
// Copyright (c) 2018 Doyub Kim
// CubbyFlow is voxel-based fluid simulation engine for computer games.
// Copyright (c) 2020 CubbyFlow Team
// Core Part: Chris Ohk, Junwoo Hwang, Jihong Sin, Seungwoo Yoo
// AI Part: Dongheon Cho, Minseo Kim
// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Core/Collider/Collider2.hpp>

namespace CubbyFlow
{
Collider2::Collider2()
{
    // Do nothing
}

Collider2::~Collider2()
{
    // Do nothing
}

void Collider2::ResolveCollision(double radius, double restitutionCoefficient,
                                 Vector2D* newPosition, Vector2D* newVelocity)
{
    assert(m_surface != nullptr);

    if (!m_surface->IsValidGeometry())
    {
        return;
    }

    ColliderQueryResult colliderPoint;

    GetClosestPoint(m_surface, *newPosition, &colliderPoint);

    // Check if the new position is penetrating the surface
    if (IsPenetrating(colliderPoint, *newPosition, radius))
    {
        // Target point is the closest non-penetrating position from the new
        // position.
        Vector2D targetNormal = colliderPoint.normal;
        Vector2D targetPoint = colliderPoint.point + radius * targetNormal;
        Vector2D colliderVelAtTargetPoint = colliderPoint.velocity;

        // Get new candidate relative velocity from the target point.
        Vector2D relativeVel = *newVelocity - colliderVelAtTargetPoint;
        double normalDotRelativeVel = targetNormal.Dot(relativeVel);
        Vector2D relativeVelN = normalDotRelativeVel * targetNormal;
        Vector2D relativeVelT = relativeVel - relativeVelN;

        // Check if the velocity is facing opposite direction of the surface
        // normal
        if (normalDotRelativeVel < 0.0)
        {
            // Apply restitution coefficient to the surface normal component of
            // the velocity
            Vector2D deltaRelativeVelN =
                (-restitutionCoefficient - 1.0) * relativeVelN;
            relativeVelN *= -restitutionCoefficient;

            // Apply friction to the tangential component of the velocity
            // From Bridson et al., Robust Treatment of Collisions,
            // Contact and Friction for Cloth Animation, 2002
            // http://graphics.stanford.edu/papers/cloth-sig02/cloth.pdf
            if (relativeVelT.LengthSquared() > 0.0)
            {
                double frictionScale = std::max(
                    1.0 - m_frictionCoeffient * deltaRelativeVelN.Length() /
                              relativeVelT.Length(),
                    0.0);
                relativeVelT *= frictionScale;
            }

            // Reassemble the components
            *newVelocity =
                relativeVelN + relativeVelT + colliderVelAtTargetPoint;
        }

        // Geometric fix
        *newPosition = targetPoint;
    }
}

double Collider2::GetFrictionCoefficient() const
{
    return m_frictionCoeffient;
}

void Collider2::SetFrictionCoefficient(double newFrictionCoeffient)
{
    m_frictionCoeffient = std::max(newFrictionCoeffient, 0.0);
}

const Surface2Ptr& Collider2::GetSurface() const
{
    return m_surface;
}

void Collider2::SetSurface(const Surface2Ptr& newSurface)
{
    m_surface = newSurface;
}

void Collider2::GetClosestPoint(const Surface2Ptr& surface,
                                const Vector2D& queryPoint,
                                ColliderQueryResult* result) const
{
    result->distance = surface->ClosestDistance(queryPoint);
    result->point = surface->ClosestPoint(queryPoint);
    result->normal = surface->ClosestNormal(queryPoint);
    result->velocity = VelocityAt(queryPoint);
}

bool Collider2::IsPenetrating(const ColliderQueryResult& colliderPoint,
                              const Vector2D& position, double radius)
{
    // If the new candidate position of the particle is inside
    // the volume defined by the surface OR the new distance to the surface is
    // less than the particle's radius, this particle is in colliding state.
    return m_surface->IsInside(position) || colliderPoint.distance < radius;
}

void Collider2::Update(double currentTimeInSeconds,
                       double timeIntervalInSeconds)
{
    assert(m_surface != nullptr);

    if (!m_surface->IsValidGeometry())
    {
        return;
    }

    m_surface->UpdateQueryEngine();

    if (m_onUpdateCallback)
    {
        m_onUpdateCallback(this, currentTimeInSeconds, timeIntervalInSeconds);
    }
}

void Collider2::SetOnBeginUpdateCallback(const OnBeginUpdateCallback& callback)
{
    m_onUpdateCallback = callback;
}
}  // namespace CubbyFlow