#include "DataStreaming.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
typedef int SOCKET;

const int BUFFER_SIZE = 2048;

WebStreamer::WebStreamer(const std::string defined_address) {
    this->address = defined_address;
    try {
        auto pc = std::make_shared<rtc::PeerConnection>();

        pc->onStateChange(
            [](rtc::PeerConnection::State state) { std::cout << "State: " << state << std::endl; });

        SOCKET sock = socket(AF_INET, SOCK_DGRAM, 0);
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        addr.sin_port = htons(6000);

        if (bind(sock, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) < 0)
            throw std::runtime_error("Failed to bind UDP socket on 127.0.0.1:6000");

        int rcvBufSize = 212992;
        setsockopt(sock, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char *>(&rcvBufSize),
                    sizeof(rcvBufSize));

        const rtc::SSRC ssrc = 42;
        rtc::Description::Video media("video", rtc::Description::Direction::SendOnly);
        media.addH264Codec(96); // Must match the payload type of the external h264 RTP stream
        media.addSSRC(ssrc, "video-send");
        auto track = pc->addTrack(media);
        pc->setLocalDescription();

		// Receive from UDP
		char buffer[BUFFER_SIZE];
		int len;
		while ((len = recv(sock, buffer, BUFFER_SIZE, 0)) >= 0) {
			if (len < sizeof(rtc::RtpHeader) || !track->isOpen())
				continue;

			auto rtp = reinterpret_cast<rtc::RtpHeader *>(buffer);
			rtp->setSsrc(ssrc);

			track->send(reinterpret_cast<const std::byte *>(buffer), len);
		}
    }
    catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
	}
}


