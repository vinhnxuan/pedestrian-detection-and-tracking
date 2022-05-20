#include "rtc/rtc.hpp"

class WebStreamer {
public:
    explicit WebStreamer(const std::string defined_address);
private:
    std::string address;
};