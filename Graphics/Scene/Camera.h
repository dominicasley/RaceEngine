#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


class Camera
{
private:
    bool dirty{};
    unsigned int iso{};
    float aspectRatio;
    float aperture{};
    float exposure{};
    float fieldOfView;
    float nearClippingPlane{};
    float farClippingPlane{};
    glm::vec3 position{};
    glm::vec3 direction;
    glm::vec3 roll;
    glm::mat4 mvp{};
    glm::mat4 mv{};

public:
    Camera();

    void setPosition(float x, float y, float z);

    void setDirection(float x, float y, float z);

    void setRoll(float x, float y, float z);

    void setAspectRatio(float aspectRatio);

    void setFieldOfView(float);

    void lookAtPoint(float, float, float);

    void update();

    [[nodiscard]] const glm::vec3 &getPosition() const;

    [[nodiscard]] const glm::mat4 &getMvpMatrix();

    [[nodiscard]] const glm::mat4 &getMvMatrix();
};

