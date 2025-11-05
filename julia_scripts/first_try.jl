using LinearAlgebra
using Printf
using ProgressMeter
using Statistics
using GLMakie

function acoustic_two_media_si()
    println("=== АКУСТИЧЕСКОЕ МОДЕЛИРОВАНИЕ В ВОЗДУХЕ И ТВЕРДОМ ТЕЛЕ ===")
    
    # ====================
    # ПАРАМЕТРЫ МОДЕЛИ (СИ)
    # ====================
    
    # Геометрические параметры [м]
    DOMAIN_WIDTH = 4.0        # м - ширина области
    DOMAIN_HEIGHT = 2.0       # м - высота области
    SOLID_HEIGHT = 0.8        # м - высота твердого тела (в нижней части)
    PML_THICKNESS = 0.5       # м - толщина PML слоя
    
    # Параметры материалов [СИ]
    # Воздух
    C_AIR = 343.0             # м/с - скорость звука в воздухе
    RHO_AIR = 1.225           # кг/м³ - плотность воздуха
    Z_AIR = RHO_AIR * C_AIR   # Па·с/м - акустический импеданс воздуха
    
    # Твердое тело (например, сталь)
    C_SOLID = 2000.0          # м/с - скорость звука в твердом теле
    RHO_SOLID = 7800.0        # кг/м³ - плотность твердого тела
    Z_SOLID = RHO_SOLID * C_SOLID  # Па·с/м - акустический импеданс твердого тела
    
    # Коэффициент отражения на границе (импедансные условия)
    REFLECTION_COEFFICIENT = (Z_SOLID - Z_AIR) / (Z_SOLID + Z_AIR)
    
    # Параметры источника [СИ]
    SOURCE_FREQUENCY = 1000.0  # Гц - частота источника
    SOURCE_AMPLITUDE = 1000.0  # Па - амплитуда давления источника
    
    # Временные параметры [СИ]
    SIMULATION_DURATION = 0.01  # с - длительность моделирования
    TIME_STEP = 5e-7           # с - шаг по времени
    
    # PML параметры [СИ]
    PML_STRENGTH = 3000.0     # коэффициент усиления демпфирования
    
    # Параметры дискретизации
    GRID_RESOLUTION = 40      # точек на метр
    NX = Int(DOMAIN_WIDTH * GRID_RESOLUTION)
    NY = Int(DOMAIN_HEIGHT * GRID_RESOLUTION)
    DX = DOMAIN_WIDTH / NX    # м - шаг по x
    DY = DOMAIN_HEIGHT / NY   # м - шаг по y
    
    # Координата границы раздела сред [м]
    INTERFACE_Y = SOLID_HEIGHT
    
    # Проверка устойчивости (Курант-Фридрихса-Леви)
    C_MAX = max(C_AIR, C_SOLID)
    CFL_NUMBER = C_MAX * TIME_STEP / min(DX, DY)
    if CFL_NUMBER > 1.0
        @warn "CFL число = $(CFL_NUMBER) > 1 - схема может быть неустойчивой!"
    end
    
    # ====================
    # ВЫВОД ПАРАМЕТРОВ
    # ====================
    
    println("Геометрия:")
    println("  Область: $(DOMAIN_WIDTH) × $(DOMAIN_HEIGHT) м")
    println("  Твердое тело: высота $(SOLID_HEIGHT) м (нижняя часть)")
    println("  Воздух: высота $(DOMAIN_HEIGHT - SOLID_HEIGHT) м (верхняя часть)")
    println("  Сетка: $NX × $NY точек ($(DX*1000) × $(DY*1000) мм на ячейку)")
    
    println("\nМатериальные параметры:")
    println("  Воздух: c=$(C_AIR) м/с, ρ=$(RHO_AIR) кг/м³, Z=$(round(Z_AIR, digits=1)) Па·с/м")
    println("  Твердое тело: c=$(C_SOLID) м/с, ρ=$(RHO_SOLID) кг/м³, Z=$(round(Z_SOLID, digits=0)) Па·с/м")
    println("  Коэффициент отражения: $(round(REFLECTION_COEFFICIENT, digits=3))")
    
    println("\nПараметры источника:")
    println("  Частота: $(SOURCE_FREQUENCY) Гц")
    println("  Амплитуда: $(SOURCE_AMPLITUDE) Па")
    
    println("\nВременные параметры:")
    println("  Длительность: $(SIMULATION_DURATION*1000) мс")
    println("  Шаг времени: $(TIME_STEP*1e6) мкс")
    println("  CFL число: $(round(CFL_NUMBER, digits=3))")
    
    # ====================
    # ИНИЦИАЛИЗАЦИЯ СЕТКИ И МАТЕРИАЛОВ
    # ====================
    
    # Координатные сетки [м]
    x = range(0, DOMAIN_WIDTH, length=NX)
    y = range(0, DOMAIN_HEIGHT, length=NY)
    
    # Поля давления [Па]
    p_curr = zeros(Float64, NX, NY)
    p_prev = zeros(Float64, NX, NY)
    
    # Массивы для параметров среды в каждой ячейке
    sound_speed = zeros(Float64, NX, NY)
    density = zeros(Float64, NX, NY)
    impedance = zeros(Float64, NX, NY)
    
    # Заполняем параметры сред
    for i in 1:NX, j in 1:NY
        if y[j] <= INTERFACE_Y
            # Твердое тело (нижняя часть)
            sound_speed[i, j] = C_SOLID
            density[i, j] = RHO_SOLID
            impedance[i, j] = Z_SOLID
        else
            # Воздух (верхняя часть)
            sound_speed[i, j] = C_AIR
            density[i, j] = RHO_AIR
            impedance[i, j] = Z_AIR
        end
    end
    
    # Поле коэффициентов демпфирования PML [1/с]
    damping = zeros(Float64, NX, NY)
    
    # Заполняем PML области (слева и справа)
    for i in 1:NX, j in 1:NY
        # Левый PML слой
        if x[i] < PML_THICKNESS
            normalized_dist = (PML_THICKNESS - x[i]) / PML_THICKNESS
            damping[i, j] = PML_STRENGTH * normalized_dist^2
        # Правый PML слой
        elseif x[i] > DOMAIN_WIDTH - PML_THICKNESS
            normalized_dist = (x[i] - (DOMAIN_WIDTH - PML_THICKNESS)) / PML_THICKNESS
            damping[i, j] = PML_STRENGTH * normalized_dist^2
        end
    end
    
    # Координаты источника [м] - в воздушной среде
    source_x = DOMAIN_WIDTH / 2
    source_y = INTERFACE_Y + (DOMAIN_HEIGHT - INTERFACE_Y) * 0.7  # Выше границы раздела
    i_source = argmin(abs.(x .- source_x))
    j_source = argmin(abs.(y .- source_y))
    
    println("\nИсточник:")
    println("  Позиция: ($(round(source_x, digits=2)), $(round(source_y, digits=2))) м")
    println("  Среда: воздух")
    
    # ====================
    # ВРЕМЕННОЙ ЦИКЛ С УЧЕТОМ ГРАНИЦЫ РАЗДЕЛА
    # ====================
    
    results = []
    max_pressure_values = []
    energy_values = []
    
    t_steps = 0:TIME_STEP:SIMULATION_DURATION
    prog = Progress(length(t_steps), 1)
    
    println("\nНачало расчета с границей раздела сред...")
    
    for (step, t) in enumerate(t_steps)
        # ИСТОЧНИК ДАВЛЕНИЯ [Па] - в воздушной среде
        if t < 0.002
            envelope = exp(-10000*(t - 0.001)^2)
            carrier = sin(2π * SOURCE_FREQUENCY * t)
            source_pressure = SOURCE_AMPLITUDE * envelope * carrier
            p_curr[i_source, j_source] += source_pressure * TIME_STEP^2
        end
        
        # ВЫЧИСЛЕНИЕ НОВОГО СОСТОЯНИЯ
        p_next = zeros(Float64, NX, NY)
        
        for i in 2:NX-1, j in 2:NY-1
            c = sound_speed[i, j]
            γ = damping[i, j]
            
            # Лапласиан давления
            d2p_dx2 = (p_curr[i+1, j] - 2*p_curr[i, j] + p_curr[i-1, j]) / DX^2
            d2p_dy2 = (p_curr[i, j+1] - 2*p_curr[i, j] + p_curr[i, j-1]) / DY^2
            laplacian = d2p_dx2 + d2p_dy2
            
            # Волновое уравнение с PML
            p_next[i, j] = (2 - γ*TIME_STEP) * p_curr[i, j] - (1 - γ*TIME_STEP) * p_prev[i, j] + (c * TIME_STEP)^2 * laplacian
        end
        
        # ====================
        # ГРАНИЧНЫЕ УСЛОВИЯ
        # ====================
        
        # 1. Верхняя граница (воздух-стенка): жесткая стенка (p = 0 Па)
        p_next[:, end] .= 0.0
        
        # 2. Нижняя граница (твердое тело-стенка): жесткая стенка (p = 0 Па)
        p_next[:, 1] .= 0.0
        
        # 3. Граница раздела сред: импедансные условия
        # Находим индексы ячеек на границе раздела
        j_interface = argmin(abs.(y .- INTERFACE_Y))
        
        # Условия непрерывности на границе раздела:
        # - Непрерывность давления: p_air = p_solid
        # - Непрерывность нормальной скорости: (1/ρ_air) ∂p/∂y|_air = (1/ρ_solid) ∂p/∂y|_solid
        
        # Для упрощения используем приближенные условия
        for i in 2:NX-1
            j = j_interface
            
            # Ячейки выше границы (воздух)
            j_air = j + 1
            # Ячейки ниже границы (твердое тело)
            j_solid = j - 1
            
            if j_air <= NY && j_solid >= 1
                # Приближенное условие сопряжения
                # Используем линейную интерполяцию для учета разницы импедансов
                p_air = p_next[i, j_air]
                p_solid = p_next[i, j_solid]
                
                # Усреднение с учетом импедансов
                p_next[i, j] = (Z_AIR * p_solid + Z_SOLID * p_air) / (Z_AIR + Z_SOLID)
            end
        end
        
        # ОБНОВЛЕНИЕ ПОЛЕЙ
        p_prev, p_curr = p_curr, p_next
        
        # ДИАГНОСТИКА
        current_max_pressure = maximum(abs.(p_curr))
        push!(max_pressure_values, current_max_pressure)
        
        # Энергия в системе [Дж/м²]
        energy = sum(p_curr.^2) * DX * DY / (mean(density) * mean(sound_speed)^2)
        push!(energy_values, energy)
        
        # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
        if step % 100 == 0
            push!(results, (t=t, p=copy(p_curr), max_pressure=current_max_pressure, energy=energy))
            ProgressMeter.next!(prog; showvalues = [
                (:time, @sprintf("%.4f с", t)),
                (:max_p, @sprintf("%.1f Па", current_max_pressure)),
                (:energy, @sprintf("%.2e Дж/м²", energy))
            ])
        end
    end
    
    # ====================
    # ФИНАЛЬНАЯ ДИАГНОСТИКА
    # ====================
    
    println("\n=== РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ ===")
    println("Максимальное давление: $(round(maximum(max_pressure_values), digits=1)) Па")
    println("Коэффициент отражения теоретический: $(round(REFLECTION_COEFFICIENT, digits=3))")
    
    return results, x, y, max_pressure_values, energy_values, damping, sound_speed, INTERFACE_Y
end


function create_two_media_animation(results, x, y, max_pressure_values, energy_values, damping, sound_speed, interface_y)
    if isempty(results)
        println("Ошибка: results пустой — анимация не может быть создана.")
        return
    end

    println("\nСоздание анимации для двух сред...")
    
    # Графики диагностики (строим один раз)
    times = [r.t for r in results]
    pressures = [r.max_pressure for r in results]
    energies = [r.energy for r in results]
    
    fig = Figure(size=(1400, 800))  # Увеличили ширину для colorbar
    
    # p1: Максимальное давление
    ax1 = Axis(fig[1,1], title="Максимальное давление", xlabel="Время, с", ylabel="Давление, Па")
    GLMakie.lines!(ax1, times, pressures, color=:blue, linewidth=2)
    
    # p2: Энергия
    ax2 = Axis(fig[1,2], title="Энергия в системе", xlabel="Время, с", ylabel="Энергия, Дж/м²")
    GLMakie.lines!(ax2, times, energies, color=:red, linewidth=2)
    
    # p3: Heatmap для давления (будем обновлять)
    ax3 = Axis(fig[2,1:2], title="Две среды: воздух/твердое тело", xlabel="x, м", ylabel="y, м", aspect=DataAspect())
    
    # Инициализируем heatmap с начальными данными (первый кадр)
    p_init = results[1].p  # Первое давление
    current_max = maximum(abs.(p_init))
    clim_range = current_max > 0 ? (-current_max, current_max) : (-1.0, 1.0)
    hm = GLMakie.heatmap!(ax3, x, y, p_init, colormap=:viridis, colorrange=clim_range)  # Без транспонирования; скорректируйте, если данные не в правильной ориентации
    
    # Colorbar в отдельной позиции (справа от ax3)
    Colorbar(fig[2,3], hm, label="Давление, Па")  # vertical=true по умолчанию
    
    # Разметка (один раз)
    GLMakie.hlines!(ax3, [interface_y], linewidth=3, color=:red, linestyle=:dash, label="Граница раздела")
    GLMakie.hlines!(ax3, [0, maximum(y)], linewidth=2, color=:white, linestyle=:solid, label="Жесткие стенки")
    GLMakie.vlines!(ax3, [0.5, 3.5], linewidth=2, color=:white, linestyle=:dash, label="PML")
    
    # Подписи сред
    GLMakie.text!(ax3, maximum(x)*0.1, interface_y/2, text="Твердое тело", fontsize=10, color=:white)
    GLMakie.text!(ax3, maximum(x)*0.1, interface_y + (maximum(y)-interface_y)/2, text="Воздух", fontsize=10, color=:white)
    
    # Анимация: обновляем только heatmap и title
    record(fig, "two_media_acoustic.mp4", enumerate(results); framerate=10) do (i, result)
        t = result.t
        p = result.p
        
        # Обновляем данные heatmap
        current_max = maximum(abs.(p))
        clim_range = current_max > 0 ? (-current_max, current_max) : (-1.0, 1.0)
        hm[3] = p  # Обновляем матрицу данных (без транспонирования; скорректируйте при необходимости)
        hm.colorrange = clim_range  # Обновляем цветовую шкалу
        
        # Обновляем title
        ax3.title = @sprintf("Две среды: воздух/твердое тело\nt = %.4f с", t)
    end
    
    println("Анимация сохранена как 'two_media_acoustic.mp4'")
    
    # Дополнительные диагностические графики (отдельные фигуры)
    # Карта скоростей звука
    fig_sound = Figure(size=(800, 600))
    ax_sound = Axis(fig_sound[1,1], title="Скорость звука в средах", xlabel="x, м", ylabel="y, м", aspect=DataAspect())
    hm_sound = GLMakie.heatmap!(ax_sound, x, y, sound_speed, colormap=:plasma)
    Colorbar(fig_sound[1,2], hm_sound, label="Скорость, м/с")  # Справа
    GLMakie.hlines!(ax_sound, [interface_y], linewidth=3, color=:red, linestyle=:dash, label="Граница раздела")
    save("sound_speed_map.png", fig_sound)
    
    # Карта PML демпфирования
    fig_damping = Figure(size=(800, 600))
    ax_damping = Axis(fig_damping[1,1], title="Коэффициенты демпфирования PML", xlabel="x, м", ylabel="y, м", aspect=DataAspect())
    hm_damping = GLMakie.heatmap!(ax_damping, x, y, damping, colormap=:hot)
    Colorbar(fig_damping[1,2], hm_damping, label="Демпфирование, 1/с")  # Справа
    save("pml_damping_two_media.png", fig_damping)
    
    println("Диагностические графики сохранены")
end


# ====================
# АНАЛИЗ ОТРАЖЕНИЯ И ПРОХОЖДЕНИЯ
# ====================

function analyze_interface_effects(results, x, y, interface_y)
    println("\n=== АНАЛИЗ ЭФФЕКТОВ НА ГРАНИЦЕ РАЗДЕЛА ===")
    
    # Находим индексы для различных областей
    j_interface = argmin(abs.(y .- interface_y))
    j_air = Int(round(j_interface + (length(y) - j_interface) / 2))  # Середина воздушной области
    j_solid = Int(round(j_interface / 2))  # Середина твердого тела
    
    i_center = length(x) ÷ 2
    
    # Извлекаем данные о давлении в разных средах
    times = [r.t for r in results]
    pressure_air = [r.p[i_center, j_air] for r in results]
    pressure_solid = [r.p[i_center, j_solid] for r in results]
    pressure_interface = [r.p[i_center, j_interface] for r in results]
    
    # Анализ коэффициента отражения
    max_air = maximum(abs.(pressure_air))
    max_solid = maximum(abs.(pressure_solid))
    
    measured_reflection = (max_air - max_solid) / (max_air + max_solid)
    
    println("Коэффициент отражения (измеренный): $(round(measured_reflection, digits=3))")
    println("Максимальное давление в воздухе: $(round(max_air, digits=1)) Па")
    println("Максимальное давление в твердом теле: $(round(max_solid, digits=1)) Па")
    
    # Графики давления в разных средах (на Makie)
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1,1], title="Давление в различных средах", xlabel="Время, с", ylabel="Давление, Па")
    
    GLMakie.lines!(ax, times, pressure_air, label="Воздух (центр)", linewidth=2)
    GLMakie.lines!(ax, times, pressure_solid, label="Твердое тело (центр)", linewidth=2)
    GLMakie.lines!(ax, times, pressure_interface, label="Граница раздела", linewidth=2, linestyle=:dash)
    
    axislegend(ax)  # Добавляем легенду
    
    save("interface_pressure.png", fig)
    
    return measured_reflection
end

# ====================
# ЗАПУСК МОДЕЛИ
# ====================

println("Запуск акустического моделирования в двух средах...")
results, x, y, max_pressure, energy, damping, sound_speed, interface_y = acoustic_two_media_si()
create_two_media_animation(results, x, y, max_pressure, energy, damping, sound_speed, interface_y)
reflection_coeff = analyze_interface_effects(results, x, y, interface_y)

println("\n=== МОДЕЛИРОВАНИЕ С ДВУМЯ СРЕДАМИ ЗАВЕРШЕНО ===")
println("Особенности модели:")
println("  - Две среды: воздух (верх) и твердое тело (низ)")
println("  - Импедансные условия на границе раздела")
println("  - Жесткие стенки сверху и снизу")
println("  - PML поглощение по бокам")
println("  - Учет разных скоростей звука и плотностей")
