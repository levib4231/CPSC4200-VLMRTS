# Compile on the host architecture; regenerate protobuf from the patched schema.
FROM mcr.microsoft.com/dotnet/sdk:8.0-bookworm-slim@sha256:1b955f179b02df4f0eb5fadedaa898cbe44d620a64aab52eec72c8e91d1355b5 AS engine
ENV DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 DOTNET_CLI_TELEMETRY_OPTOUT=1
RUN apt-get update && apt-get install -y --no-install-recommends git make ca-certificates libsdl2-dev libopenal-dev libfreetype-dev liblua5.1-0-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src/openra
ARG OPENRA_REF=8a5d224223e0498e006a7350a9767a87bd45a708
RUN git init && git remote add origin https://github.com/yxc20089/OpenRA.git && git fetch --depth 1 origin "$OPENRA_REF" && git checkout --detach FETCH_HEAD
RUN dotnet restore OpenRA.sln
COPY server_extensions/engine-team.patch /tmp/engine-team.patch
RUN git apply --check /tmp/engine-team.patch && git apply /tmp/engine-team.patch
# Grpc.Tools 2.69's Linux ARM64 protoc crashes. Use Debian's native protoc
# and the matching native C# plugin, rather than emulating the .NET SDK.
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler && rm -rf /var/lib/apt/lists/*
RUN ARCH=$( [ "$(dpkg --print-architecture)" = "arm64" ] && echo arm64 || echo x64 ) && \
    dotnet build -c Release -nologo -p:TargetPlatform=unix-generic \
    -p:Protobuf_ProtocFullPath=/usr/bin/protoc \
    -p:gRPC_PluginFullPath=/root/.nuget/packages/grpc.tools/2.69.0/tools/linux_${ARCH}/grpc_csharp_plugin \
    && ./configure-system-libraries.sh

FROM ghcr.io/yxc20089/openra-rl@sha256:a15619e9dc4b3ccb5257d363d5b657673a3c1f7971a731cf2e3c176b7c4aae2c
COPY --from=engine /src/openra/bin/ /opt/openra/bin/
COPY --from=engine /src/openra/mods/ /opt/openra/mods/
COPY --from=engine /src/openra/glsl/ /opt/openra/glsl/
COPY maps/vlm-team-doubles.oramap /opt/openra/mods/ra/maps/vlm-team-doubles.oramap
RUN LIBDIR=$( [ "$(dpkg --print-architecture)" = "arm64" ] && echo /usr/lib/aarch64-linux-gnu || echo /usr/lib/x86_64-linux-gnu ) && \
    ln -sf "$LIBDIR/libSDL2-2.0.so.0" /opt/openra/bin/SDL2.so && \
    ln -sf "$LIBDIR/libopenal.so.1" /opt/openra/bin/soft_oal.so && \
    ln -sf "$LIBDIR/libfreetype.so.6" /opt/openra/bin/freetype6.so && \
    ln -sf "$LIBDIR/liblua5.1.so.0" /opt/openra/bin/lua51.so
ENV RL_CONTROLLED_PLAYERS=Multi1,Multi2
HEALTHCHECK NONE
EXPOSE 9999
ENTRYPOINT ["dotnet", "/opt/openra/bin/OpenRA.dll", "Engine.EngineDir=/opt/openra", "Game.Mod=ra", "Game.Platform=Null"]
CMD ["Launch.Map=vlm-team-doubles.oramap", "Launch.Bots=Multi1:rl-agent,Multi2:dummy,Multi0:normal,Multi3:normal", "Server.RecordReplays=True"]
