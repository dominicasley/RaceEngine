module;

#include <spdlog/logger.h>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include "PxPhysicsAPI.h"

export module Physics;

export struct PhysicsScene {
    physx::PxScene* physxScene = nullptr;
};

export class PhysicsService {
    physx::PxFoundation* foundation;
    physx::PxPhysics* physics;
    physx::PxDefaultErrorCallback defaultErrorCallback;
    physx::PxDefaultAllocator defaultAllocatorCallback;
    physx::PxDefaultCpuDispatcher* dispatcher;

    std::vector<PhysicsScene> scenes;

public:
    explicit PhysicsService(spdlog::logger& logger) {
        foundation = PxCreateFoundation(PX_PHYSICS_VERSION, defaultAllocatorCallback, defaultErrorCallback);

        if(!foundation)
            logger.error("PxCreateFoundation failed!");

        physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, physx::PxTolerancesScale(), true, nullptr);

        if(!physics)
            logger.error("PxCreatePhysics failed!");
    }

    ~PhysicsService() {
        for (auto& scene : scenes) {
            scene.physxScene->release();
        }

        physics->release();
        foundation->release();
    }

    PhysicsScene& createScene(glm::vec3 gravity) {
        auto sceneDescription = physx::PxSceneDesc(physics->getTolerancesScale());

        sceneDescription.gravity = physx::PxVec3(gravity.x, gravity.y, gravity.z);
        dispatcher = physx::PxDefaultCpuDispatcherCreate(2);
        sceneDescription.cpuDispatcher	= dispatcher;
        sceneDescription.filterShader	= physx::PxDefaultSimulationFilterShader;

        return scenes.emplace_back(
            PhysicsScene {
                .physxScene = physics->createScene(sceneDescription)
            }
        );
    }

    void step() {
        for (auto& scene : scenes) {
            scene.physxScene->simulate(1.0f/60.0f);
            scene.physxScene->fetchResults();
        }
    }

    void createGroundPlane(PhysicsScene& scene, glm::vec4 normal) {
        auto material = physics->createMaterial(0.5f, 0.5f, 0.6f);
        auto plane = physx::PxCreatePlane(*physics, physx::PxPlane(normal.x, normal.y, normal.z, normal.w), *material);

        scene.physxScene->addActor(*plane);
    }
};

