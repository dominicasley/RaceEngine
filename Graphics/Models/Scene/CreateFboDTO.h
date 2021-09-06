#pragma once

#include <vector>

#include "FboType.h"
#include "CreateFboAttachmentDTO.h"

struct CreateFboDTO
{
    FboType type;
    std::vector<CreateFboAttachmentDTO> attachments;
};